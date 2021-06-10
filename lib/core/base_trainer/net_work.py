# -*-coding:utf-8-*-

import sklearn.metrics
import cv2
import time
import os
import pandas as pd
from torch.utils.data import DataLoader

from lib.core.utils.torch_utils import EMA
from lib.dataset.dataietr import AlaskaDataIter

from train_config import config as cfg
# from lib.dataset.dataietr import DataIter


from lib.core.base_trainer.metric import *
import torch
import torch.nn.functional as F
from lib.core.model.mix.mix import mixup,mixup_criterion
from torchcontrib.optim import SWA




from lib.core.base_trainer.model import Net


from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
torch.distributed.init_process_group(backend="nccl")

if cfg.TRAIN.mix_precision:
    from apex import amp


class Train(object):
    """Train class.
    """

    def __init__(self,
                 train_df,
                 val_df,
                 fold):

        self.train_generator = AlaskaDataIter(train_df, training_flag=True, shuffle=False)
        self.train_ds = DataLoader(self.train_generator,
                                   cfg.TRAIN.batch_size,
                                   num_workers=cfg.TRAIN.process_num,
                                   sampler=DistributedSampler(self.train_generator,
                                                              shuffle=True))

        self.val_generator = AlaskaDataIter(val_df, training_flag=False, shuffle=False)

        self.val_ds = DataLoader(self.val_generator,
                                 cfg.TRAIN.validatiojn_batch_size,
                                 num_workers=cfg.TRAIN.process_num,
                                 sampler=DistributedSampler(self.val_generator,
                                                            shuffle=False))

        self.fold = fold

        self.init_lr = cfg.TRAIN.init_lr
        self.warup_step = cfg.TRAIN.warmup_step
        self.epochs = cfg.TRAIN.epoch
        self.batch_size = cfg.TRAIN.batch_size
        self.l2_regularization = cfg.TRAIN.weight_decay_factor

        self.early_stop = cfg.MODEL.early_stop

        self.accumulation_step = cfg.TRAIN.accumulation_batch_size // cfg.TRAIN.batch_size
        # self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.gradient_clip = cfg.TRAIN.gradient_clip

        self.save_dir=cfg.MODEL.model_path
        #### make the device

        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)


        self.model = Net().to(self.device)

        self.load_weight()

        if 'Adamw' in cfg.TRAIN.opt:

            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.init_lr, eps=1.e-5,
                                               weight_decay=self.l2_regularization)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.init_lr,
                                             momentum=0.9,
                                             weight_decay=self.l2_regularization)

        if cfg.TRAIN.SWA > 0:
            ##use swa
            self.optimizer = SWA(self.optimizer)

        if cfg.TRAIN.mix_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                               device_ids=[local_rank],
                                                               output_device=local_rank,
                                                               find_unused_parameters=True)

        self.ema = EMA(self.model, 0.97)

        self.ema.register()
        ###control vars
        self.iter_num = 0

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='max', patience=5,
        #                                                             min_lr=1e-6,factor=0.5,verbose=True)

        if cfg.TRAIN.lr_scheduler=='cos':
            logger.info('lr_scheduler.CosineAnnealingLR')
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        self.epochs,
                                                                        eta_min=1.e-7)
        else:
            logger.info('lr_scheduler.ReduceLROnPlateau')
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='max',
                                                                        patience=5,
                                                                        min_lr=1e-7,
                                                                        factor=cfg.TRAIN.lr_scheduler_factor,
                                                                        verbose=True)

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)




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

                if epoch_num < 10:
                    ###excute warm up in the first epoch
                    if self.warup_step > 0:
                        if self.iter_num < self.warup_step:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                                lr = param_group['lr']

                            logger.info('warm up with learning rate: [%f]' % (lr))

                start = time.time()

                data = images.to(self.device).float()
                label = label.to(self.device).float()

                batch_size = data.shape[0]
                if np.random.uniform(0,1) < cfg.TRAIN.mixup:
                    mixed_data, mix_target = mixup(data, label,alpha=1.)
                    outputs =self.model(mixed_data)
                    current_loss = mixup_criterion(outputs,mix_target,self.criterion)
                else:
                    predictions = self.model(data)



                    current_loss = self.criterion(predictions, label)

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
                self.iter_num += 1
                time_cost_per_batch = time.time() - start

                images_per_sec = cfg.TRAIN.batch_size / time_cost_per_batch

                if self.iter_num % cfg.TRAIN.log_interval == 0:
                    log_message = '[fold %d], ' \
                                  'Train Step %d, ' \
                                  'summary_loss: %.6f, ' \
                                  'time: %.6f, ' \
                                  'speed %d images/persec' % (
                                      self.fold,
                                      self.iter_num,
                                      summary_loss.avg,
                                      time.time() - start,
                                      images_per_sec)
                    logger.info(log_message)

            if cfg.TRAIN.SWA > 0 and epoch_num >= cfg.TRAIN.SWA:
                self.optimizer.update_swa()

            return summary_loss

        def distributed_test_epoch(epoch_num):

            ROCAUC_score=ROCAUCMeter()
            summary_loss = AverageMeter()
            self.model.eval()

            with torch.no_grad():
                for  (images, labels) in tqdm(self.val_ds):


                    data = images.to(self.device).float()
                    labels = labels.to(self.device).float()
                    batch_size = data.shape[0]

                    predictions = self.model(data)


                    current_loss = self.criterion(predictions, labels)

                    ROCAUC_score.update(labels,predictions)
                    summary_loss.update(current_loss.detach().item(), batch_size)

            return ROCAUC_score, summary_loss



        best_distance = 0.
        not_improvement = 0
        for epoch in range(self.epochs):

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
            logger.info('learning rate: [%f]' % (lr))
            t = time.time()

            summary_loss = distributed_train_epoch(epoch)
            train_epoch_log_message = '[fold %d], ' \
                                      '[RESULT]: TRAIN. Epoch: %d,' \
                                      ' summary_loss: %.5f,' \
                                      ' time:%.5f' % (
                                          self.fold,
                                          epoch,
                                          summary_loss.avg,
                                          (time.time() - t))
            logger.info(train_epoch_log_message)

            if cfg.TRAIN.SWA > 0 and epoch >= cfg.TRAIN.SWA:
                ###switch to avg model
                self.optimizer.swap_swa_sgd()

            ##switch eam weighta
            if cfg.MODEL.ema:
                self.ema.apply_shadow()

            if epoch % cfg.TRAIN.test_interval == 0:
                roc_auc_score, summary_loss = distributed_test_epoch(epoch)

                val_epoch_log_message = '[fold %d], ' \
                                        '[RESULT]: VAL. Epoch: %d,' \
                                        ' val_loss: %.5f,' \
                                        ' val_roc_auc: %.5f,' \
                                        ' time:%.5f' % (
                                            self.fold,
                                            epoch,
                                            summary_loss.avg,
                                            roc_auc_score.avg,
                                            (time.time() - t))
                logger.info(val_epoch_log_message)



            if cfg.TRAIN.lr_scheduler=='cos':
                self.scheduler.step()
            else:
                self.scheduler.step(roc_auc_score.avg)

            #### save model
            if not os.access(cfg.MODEL.model_path, os.F_OK):
                os.mkdir(cfg.MODEL.model_path)
            ###save the best auc model

            #### save the model every end of epoch
            current_model_saved_name = self.save_dir+'/fold%d_epoch_%d_val_rocauc_%.6f_loss_%.6f.pth' % (self.fold,
                                                                                                epoch,
                                                                                                roc_auc_score.avg,
                                                                                                summary_loss.avg)

            logger.info('A model saved to %s' % current_model_saved_name)
            torch.save(self.model.module.state_dict(), current_model_saved_name)

            ####switch back
            if cfg.MODEL.ema:
                self.ema.restore()

            # save_checkpoint({
            #           'state_dict': self.model.state_dict(),
            #           },iters=epoch,tag=current_model_saved_name)

            if cfg.TRAIN.SWA > 0 and epoch > cfg.TRAIN.SWA:
                ###switch back to plain model to train next epoch
                self.optimizer.swap_swa_sgd()

            if roc_auc_score.avg > best_distance:
                best_distance = roc_auc_score.avg
                logger.info(' best metric score update as %.6f' % (best_distance))
                not_improvement=0
            else:
                not_improvement += 1

            if not_improvement >= self.early_stop:
                logger.info(' best metric score not improvement for %d, break' % (self.early_stop))
                break

            torch.cuda.empty_cache()

    def load_weight(self):
        if cfg.MODEL.pretrained_model is not None:
            state_dict = torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)



