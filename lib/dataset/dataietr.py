

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


                                    A.ShiftScaleRotate(shift_limit=0.1,
                                                       scale_limit=0.1,
                                                       rotate_limit=0,
                                                       p=0.5),

                                    A.HorizontalFlip(p=0.5),
                                    #A.VerticalFlip(p=0.5),



                              ])



        self.resize_trans=A.Resize(height=512, width=512)


    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):


        return len(self.df)


    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here

        fname = dp['file_path']
        label = dp['target']

        img = np.load(fname).astype(np.float32)[[0,2,4]]  # shape: (3, 273, 256)

        ## to hwc
        img = np.transpose(img,axes=[1,2,0])

        if is_training:

            transformed = self.train_trans(image=img)
            img = transformed['image']

        ## to chw
        img = np.transpose(img, axes=[2,0,1])

        img = np.vstack(img)  # shape: (819, 256)

        transformed = self.resize_trans(image=img)
        img = transformed['image']

        img = np.expand_dims(img, 0)
        label = np.array(label,dtype=np.int)
        label =np.expand_dims(label,0)
        return img,label
