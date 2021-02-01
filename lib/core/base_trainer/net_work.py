#-*-coding:utf-8-*-

import sklearn.metrics
import cv2
import time
import os


from lib.core.utils.torch_utils import EMA
from train_config import config as cfg
#from lib.dataset.dataietr import DataIter

import sklearn.metrics
from lib.helper.logger import logger
from lib.core.model.loss.focal_loss import FocalLoss,FocalLoss4d
from lib.core.base_trainer.model import Net

import random
from lib.core.model.loss.labelsmooth import LabelSmoothing
from lib.core.model.loss.ohem import OHEMLoss
from lib.core.model.loss.labelsmooth import BCEWithLogitsLoss
from lib.core.model.loss.boost_loss import SoftBootstrappingLoss
from lib.core.base_trainer.metric import *
import torch
import torch.nn.functional as F
from lib.core.model.mix.fmix import FMix
from torchcontrib.optim import SWA


from lib.core.model.mix.mix import cutmix,cutmix_criterion,mixup,mixup_criterion



if cfg.TRAIN.mix_precision:
    from apex import amp

class Train(object):
  """Train class.
  """

  def __init__(self,train_ds,val_ds,fold):
    self.fold=fold

    self.init_lr=cfg.TRAIN.init_lr
    self.warup_step=cfg.TRAIN.warmup_step
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor

    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    self.model = Net().to(self.device)

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

    if cfg.TRAIN.num_gpu>1:
        self.model=nn.DataParallel(self.model)

    self.ema = EMA(self.model, 0.97)

    self.ema.register()
    ###control vars
    self.iter_num=0

    self.train_ds=train_ds

    self.val_ds = val_ds

    # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max', patience=5,
    #                                                             min_lr=1e-6,factor=0.5,verbose=True)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, self.epochs,eta_min=1.e-6)

    self.criterion = SoftBootstrappingLoss().to(self.device)

    self.criterion_val = LabelSmoothing(smoothing=0.0).to(self.device)




    self.fmix=FMix(loss_function=self.criterion)

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
      acc_score= ACCMeter()
      self.model.train()

      if cfg.MODEL.freeze_bn:
          for m in self.model.modules():
              if isinstance(m, nn.BatchNorm2d):
                  m.eval()
                  if cfg.MODEL.freeze_bn_affine:
                      m.weight.requires_grad = False
                      m.bias.requires_grad = False
      for step in range(self.train_ds.size):

        if epoch_num<10:
            ###excute warm up in the first epoch
            if self.warup_step>0:
                if self.iter_num < self.warup_step:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                        lr = param_group['lr']

                    logger.info('warm up with learning rate: [%f]' % (lr))

        start=time.time()

        images, target = self.train_ds()

        data = torch.from_numpy(images).to(self.device).float()
        target = torch.from_numpy(target).to(self.device).long()

        batch_size = data.shape[0]



        rand_dice=random.uniform(0,1)
        if rand_dice<cfg.MODEL.mixup:
            mixued_data,mixued_target=mixup(data,target,0.5)

            output = self.model(mixued_data)
            current_loss = mixup_criterion(output, mixued_target,self.criterion)
        elif rand_dice>cfg.MODEL.mixup and rand_dice<cfg.MODEL.mixup+cfg.MODEL.fmix:
            mixued_data = self.fmix(data)

            output = self.model(mixued_data)
            current_loss = self.fmix.loss(output, target)

        else:
            output = self.model(data)

            current_loss = self.criterion(output, target)

        summary_loss.update(current_loss.detach().item(), batch_size)
        acc_score.update(target,output)
        self.optimizer.zero_grad()

        if cfg.TRAIN.mix_precision:
            with amp.scale_loss(current_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            current_loss.backward()

        self.optimizer.step()
        if cfg.MODEL.ema:
            self.ema.update()
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


        if self.iter_num%cfg.TRAIN.log_interval==0:

            log_message = '[fold %d], '\
                          'Train Step %d, ' \
                          'summary_loss: %.6f, ' \
                          'accuracy: %.6f, ' \
                          'time: %.6f, '\
                          'speed %d images/persec'% (
                              self.fold,
                              self.iter_num,
                              summary_loss.avg,
                              acc_score.avg,
                              time.time() - start,
                              images_per_sec)
            logger.info(log_message)



      if cfg.TRAIN.SWA>0 and epoch_num>=cfg.TRAIN.SWA:
        self.optimizer.update_swa()

      return summary_loss,acc_score
    def distributed_test_epoch(epoch_num):
        summary_loss = AverageMeter()
        acc_score= ACCMeter()

        self.model.eval()
        t = time.time()
        with torch.no_grad():
            for step in range(self.val_ds.size):
                images, target = self.val_ds()
                data = torch.from_numpy(images).to(self.device).float()
                target = torch.from_numpy(target).to(self.device).long()
                batch_size = data.shape[0]


                output = self.model(data)
                loss = self.criterion_val(output, target)

                summary_loss.update(loss.detach().item(), batch_size)
                acc_score.update(target, output)

                if step % cfg.TRAIN.log_interval == 0:

                    log_message = '[fold %d], '\
                                  'Val Step %d, ' \
                                  'summary_loss: %.6f, ' \
                                  'acc: %.6f, ' \
                                  'time: %.6f' % (
                                  self.fold,step, summary_loss.avg, acc_score.avg, time.time() - t)

                    logger.info(log_message)


        return summary_loss,acc_score

    for epoch in range(self.epochs):

      for param_group in self.optimizer.param_groups:
        lr=param_group['lr']
      logger.info('learning rate: [%f]' %(lr))
      t=time.time()

      summary_loss,acc_score = distributed_train_epoch(epoch)

      train_epoch_log_message = '[fold %d], '\
                                '[RESULT]: Train. Epoch: %d,' \
                                ' summary_loss: %.5f,' \
                                ' acuracy: %.5f,' \
                                ' time:%.5f' % (
                                self.fold,epoch, summary_loss.avg,acc_score.avg, (time.time() - t))
      logger.info(train_epoch_log_message)

      if cfg.TRAIN.SWA > 0 and epoch >=cfg.TRAIN.SWA:

          ###switch to avg model
          self.optimizer.swap_swa_sgd()

      ##switch eam weighta
      if cfg.MODEL.ema:
        self.ema.apply_shadow()

      if epoch%cfg.TRAIN.test_interval==0:

          summary_loss,acc_score = distributed_test_epoch(epoch)

          val_epoch_log_message = '[fold %d], '\
                                  '[RESULT]: VAL. Epoch: %d,' \
                                  ' summary_loss: %.5f,' \
                                  ' accuracy: %.5f,' \
                                  ' time:%.5f' % (
                                   self.fold,epoch, summary_loss.avg,acc_score.avg, (time.time() - t))
          logger.info(val_epoch_log_message)

      self.scheduler.step()
      # self.scheduler.step(acc_score.avg)

      #### save model
      if not os.access(cfg.MODEL.model_path, os.F_OK):
          os.mkdir(cfg.MODEL.model_path)
      ###save the best auc model

      #### save the model every end of epoch
      current_model_saved_name='./models/fold%d_epoch_%d_val_loss%.6f.pth'%(self.fold,epoch,summary_loss.avg)

      logger.info('A model saved to %s' % current_model_saved_name)
      torch.save(self.model.state_dict(),current_model_saved_name)

      ####switch back
      if cfg.MODEL.ema:
        self.ema.restore()

      # save_checkpoint({
      #           'state_dict': self.model.state_dict(),
      #           },iters=epoch,tag=current_model_saved_name)

      if cfg.TRAIN.SWA > 0 and epoch > cfg.TRAIN.SWA:
          ###switch back to plain model to train next epoch
          self.optimizer.swap_swa_sgd()




  def load_weight(self):
      if cfg.MODEL.pretrained_model is not None:
          state_dict=torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
          self.model.load_state_dict(state_dict,strict=False)



