#-*-coding:utf-8-*-

import sklearn.metrics
import cv2
import time
import os
import pandas as pd
from torch.utils.data import DataLoader

from lib.core.utils.torch_utils import EMA
from lib.dataset.dataietr import AlaskaDataIter

from train_config import config as cfg
#from lib.dataset.dataietr import DataIter

import sklearn.metrics
from lib.utils.logger import logger
from lib.core.model.loss.focal_loss import FocalLoss,FocalLoss4d
from lib.core.base_trainer.model import  Encoder,DecoderWithAttention

import random
from lib.core.model.loss.labelsmooth import LabelSmoothing
from lib.core.model.loss.ohem import OHEMLoss
from lib.core.model.loss.labelsmooth import BCEWithLogitsLoss

from lib.core.base_trainer.metric import *
import torch
import torch.nn.functional as F
from lib.core.model.mix.fmix import FMix
from torchcontrib.optim import SWA


from lib.core.model.mix.mix import cutmix,cutmix_criterion,mixup,mixup_criterion
from lib.core.base_trainer.model import Caption


if cfg.TRAIN.mix_precision:
    from apex import amp

class Train(object):
  """Train class.
  """

  def __init__(self,
               train_df,
               val_df,
               fold,
               tokenizer):

    self.train_generator = AlaskaDataIter(train_df, tokenizer, training_flag=True, shuffle=False)
    self.train_ds = DataLoader(self.train_generator,
                            cfg.TRAIN.batch_size,
                            num_workers=cfg.TRAIN.process_num,
                            shuffle=True)

    self.val_generator = AlaskaDataIter(val_df, tokenizer, training_flag=False, shuffle=False)
    self.val_ds = DataLoader(self.val_generator,
                           cfg.TRAIN.batch_size,
                           num_workers=cfg.TRAIN.process_num,
                           shuffle=False)



    self.fold=fold

    self.init_lr=cfg.TRAIN.init_lr
    self.warup_step=cfg.TRAIN.warmup_step
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor

    self.early_stop=cfg.MODEL.early_stop

    self.accumulation_step=cfg.TRAIN.accumulation_batch_size//cfg.TRAIN.batch_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    self.word_tool = tokenizer
    self.train_generate_length=cfg.MODEL.train_length

    self.gradient_clip=cfg.TRAIN.gradient_clip

    embed_dim = 200

    attention_dim = 300
    encoder_dim = 512
    decoder_dim = 300

    self.model=Caption(embed_dim=embed_dim,
                       vocab_size=len(self.word_tool),
                       attention_dim=attention_dim,
                       encoder_dim=encoder_dim,
                       decoder_dim=decoder_dim,
                       dropout=0.5,
                       max_length=cfg.MODEL.train_length,
                       tokenizer=self.word_tool).to(self.device)


    self.load_weight()



    if 'Adamw' in cfg.TRAIN.opt:

      self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=self.init_lr,eps=1.e-5,
                                         weight_decay=self.l2_regularization)
    else:
      self.optimizer = torch.optim.SGD(self.model.parameters(),
                                       lr=self.init_lr,
                                       momentum=0.9,
                                       weight_decay=self.l2_regularization)

    if cfg.TRAIN.SWA>0:
        ##use swa
        self.optimizer = SWA(self.optimizer)

    if cfg.TRAIN.mix_precision:
        self.model, self.optimizer = amp.initialize( self.model, self.optimizer, opt_level="O1")


    self.model=nn.DataParallel(self.model)

    self.ema = EMA(self.model, 0.97)

    self.ema.register()
    ###control vars
    self.iter_num=0


    # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max', patience=5,
    #                                                             min_lr=1e-6,factor=0.5,verbose=True)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, self.epochs,eta_min=1.e-7)

    self.criterion = nn.CrossEntropyLoss(ignore_index=self.word_tool.stoi["<pad>"]).to(self.device)


    self.criterion_val = nn.CrossEntropyLoss().to(self.device)


    self.fmix=FMix(loss_function=self.criterion,size=(cfg.MODEL.height,cfg.MODEL.width))

  def custom_loop(self):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(epoch_num):

      summary_loss = AverageMeter()

      self.model.train()

      if cfg.MODEL.freeze_bn:
          for m in self.model.modules():
              if isinstance(m, nn.BatchNorm2d):
                  m.eval()
                  if cfg.MODEL.freeze_bn_affine:
                      m.weight.requires_grad = False
                      m.bias.requires_grad = False

      for images, label in self.train_ds:

        if epoch_num<10:
            ###excute warm up in the first epoch
            if self.warup_step>0:
                if self.iter_num < self.warup_step:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                        lr = param_group['lr']

                    logger.info('warm up with learning rate: [%f]' % (lr))

        start=time.time()

        data = images.to(self.device).float()
        label = label.to(self.device).long()

        batch_size = data.shape[0]

        predictions, alpha = self.model(data,label,self.train_generate_length-1)

        predictions=predictions.reshape(-1,len(self.word_tool))
        target=label[:,1:].reshape(-1)

        current_loss = self.criterion(predictions,target )

        summary_loss.update(current_loss.detach().item(), batch_size)



        if cfg.TRAIN.mix_precision:
            with amp.scale_loss(current_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            current_loss.backward()
        
        if ((self.iter_num + 1) % self.accumulation_step) == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip, norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()
        if cfg.MODEL.ema:
            self.ema.update()
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


        if self.iter_num%cfg.TRAIN.log_interval==0:




            log_message = '[fold %d], '\
                          'Train Step %d, ' \
                          'summary_loss: %.6f, ' \
                          'time: %.6f, '\
                          'speed %d images/persec'% (
                              self.fold,
                              self.iter_num,
                              summary_loss.avg,
                              time.time() - start,
                              images_per_sec)
            logger.info(log_message)


      if cfg.TRAIN.SWA>0 and epoch_num>=cfg.TRAIN.SWA:
        self.optimizer.update_swa()

      return summary_loss
    def distributed_test_epoch(epoch_num):

        L_distance_meter=DISTANCEMeter()

        self.model.eval()
        t = time.time()

        text_preds = []
        with torch.no_grad():
            for step,(images) in enumerate(self.val_ds):

                data = images.to(self.device).float()

                batch_size = data.shape[0]

                predictions = self.model(data)
                predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
                
                _text_preds = self.word_tool.predict_captions(predicted_sequence)
                text_preds+=_text_preds

        text_preds = [f"InChI=1S/{text}" for text in text_preds]

        L_distance_meter.update(self.val_generator.df['InChI'].values,
                                text_preds)



        return L_distance_meter




    best_distance=1000.
    not_improvement=0
    for epoch in range(self.epochs):

      for param_group in self.optimizer.param_groups:
        lr=param_group['lr']
      logger.info('learning rate: [%f]' %(lr))
      t=time.time()

      summary_loss = distributed_train_epoch(epoch)
      train_epoch_log_message = '[fold %d], '\
                                '[RESULT]: TRAIN. Epoch: %d,' \
                                ' summary_loss: %.5f,' \
                                ' time:%.5f' % (
                                self.fold,
                                epoch,
                                summary_loss.avg,
                                (time.time() - t))
      logger.info(train_epoch_log_message)

      if cfg.TRAIN.SWA > 0 and epoch >=cfg.TRAIN.SWA:

          ###switch to avg model
          self.optimizer.swap_swa_sgd()

      ##switch eam weighta
      if cfg.MODEL.ema:
        self.ema.apply_shadow()

      if epoch%cfg.TRAIN.test_interval==0:

          distance_meter = distributed_test_epoch(epoch)

          val_epoch_log_message = '[fold %d], '\
                                  '[RESULT]: VAL. Epoch: %d,' \
                                  ' L_distance: %.5f,' \
                                  ' time:%.5f' % (
                                   self.fold,
                                   epoch,
                                   distance_meter.avg,
                                   (time.time() - t))
          logger.info(val_epoch_log_message)

      self.scheduler.step()
      # self.scheduler.step(acc_score.avg)

      #### save model
      if not os.access(cfg.MODEL.model_path, os.F_OK):
          os.mkdir(cfg.MODEL.model_path)
      ###save the best auc model

      #### save the model every end of epoch
      current_model_saved_name='./models/fold%d_epoch_%d_val_dis_%.6f.pth'%(self.fold,
                                                                                         epoch,
                                                                                         distance_meter.avg)

      logger.info('A model saved to %s' % current_model_saved_name)
      torch.save(self.model.module.state_dict(),current_model_saved_name)

      ####switch back
      if cfg.MODEL.ema:
        self.ema.restore()

      # save_checkpoint({
      #           'state_dict': self.model.state_dict(),
      #           },iters=epoch,tag=current_model_saved_name)

      if cfg.TRAIN.SWA > 0 and epoch > cfg.TRAIN.SWA:
          ###switch back to plain model to train next epoch
          self.optimizer.swap_swa_sgd()

      if distance_meter.avg<best_distance:
          best_distance=distance_meter.avg
          logger.info(' best metric score update as %.6f' % (best_distance))
      else:
          not_improvement+=1

      if not_improvement>=self.early_stop:
          logger.info(' best metric score not improvement for %d, break'%(self.early_stop))
          break

      torch.cuda.empty_cache()

  def load_weight(self):
      if cfg.MODEL.pretrained_model is not None:
          state_dict=torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
          self.model.load_state_dict(state_dict,strict=False)



