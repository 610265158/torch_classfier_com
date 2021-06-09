

import random
import cv2
import json
import numpy as np
import copy

from lib.utils.logger import logger
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
        logger.info(' contains%d samples'%len(self.df))
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



        label = np.array(label,dtype=np.int)
        label =np.expand_dims(label,0)
        return img,label
