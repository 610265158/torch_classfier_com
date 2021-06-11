

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
        logger.info(' contains%d samples  %d pos' %( len(self.df), np.sum(self.df['target']==1)))


        #
        logger.info(' contains%d samples'%len(self.df))
        self.train_trans=A.Compose([
                                    A.CoarseDropout(max_width=32,max_height=32,max_holes=8),
                                    A.ShiftScaleRotate(shift_limit=0.1,
                                                       scale_limit=0.1,
                                                       rotate_limit=0,
                                                       p=0.8)

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



        img = np.load(fname).astype(np.float32)
        img = np.transpose(img,[0,2,1])


        img = np.transpose(img,[1,2,0])

        if is_training:
            transformed = self.train_trans(image=img)

            img = transformed['image']

        img = np.transpose(img, [2,0,1])

        label = np.array(label,dtype=np.int)
        label =np.expand_dims(label,0)
        return img,label
