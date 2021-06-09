

import random
import cv2
import json
import numpy as np
import copy

from lib.utils.logger import logger
from tensorpack.dataflow import DataFromGenerator, BatchData, MultiProcessPrefetchData, PrefetchDataZMQ, RepeatedData
import time

import traceback
from lib.dataset.augmentor.augmentation import Rotate_aug,\
                                                Affine_aug,\
                                                Mirror,\
                                                Padding_aug,\
                                                Img_dropout,Random_crop



from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter

from train_config import config as cfg
import albumentations as A
import os





class AlaskaDataIter():
    def __init__(self, df,
                 training_flag=True,shuffle=True):



        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None     ##decided by self.parse_file


        self.df=df
        #
        # logger.info('after balance contains%d samples'%len(self.lst))
        self.train_trans=A.Compose([   A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width)
                              ])


        self.val_trans=A.Compose([

                                   A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width)

                                  ])


    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):


        return len(self.df)




    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here

        fname = dp['file_path']
        label = dp['target']



        img = np.load(fname)[[0, 2, 4]]  # shape: (3, 273, 256)
        img = np.stack(img,axis=0)

        print(img.shape)

        label = np.array(label,dtype=np.int)
        return img,label
