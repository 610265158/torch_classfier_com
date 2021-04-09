

import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
from lib.utils.logger import logger


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
    def __init__(self, df,tokenizer,
                 training_flag=True,shuffle=True):



        self.training_flag = training_flag
        self.shuffle = shuffle

        self.df=df
        logger.info('contains %d samples'%len(self.df))
        self.word_tool=tokenizer




        self.train_trans=A.Compose([A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width),



                              ] ,

                              )


        self.val_trans=A.Compose([

                                   A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width)

                                  ])


    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):


        return len(self.df)



    def cracy_rotate(self, image, block_nums=2):


        pitches=[]
        block_size = 384 // block_nums
        for i in range(block_nums):
            for j in range(block_nums):
                cur_pitch = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, :]


                if random.uniform(0, 1) >= 0.5:
                    random_angle = [0,1,2,3]
                    cur_pitch = np.rot90(cur_pitch, random.choice(random_angle))

                pitches.append(cur_pitch)
        random.shuffle(pitches)

        cnt=0
        for i in range(block_nums):
            for j in range(block_nums):

                image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, :] = pitches[cnt]

                cnt+=1

        return image




    def random_resize(self,image,scale_range=[0.5,1]):

        scale=random.uniform(*scale_range)

        resize_method=random.choice([cv2.INTER_NEAREST,
                                     cv2.INTER_LINEAR,
                                     cv2.INTER_CUBIC,
                                     cv2.INTER_AREA])

        image_resized=cv2.resize(image,dsize=None,fx=scale,fy=scale,interpolation=resize_method)

        return image_resized

    def addSaltNoise(self,image,SNR=0.99):

        if SNR==1:
            return image

        h,w = image.shape

        noiseSize = int(h*w * (1 - SNR))

        for k in range(0, noiseSize):

            xi = int(np.random.uniform(0, image.shape[1]))
            xj = int(np.random.uniform(0, image.shape[0]))

            image[xj, xi] = 0

        return image

    def letterbox(self, image, target_shape=(cfg.MODEL.height, cfg.MODEL.width)):

        h, w = image.shape

        if h / w >= target_shape[0] / target_shape[1]:
            size = (h, int((target_shape[1]/target_shape[0]) * h))

            image_container = np.zeros(shape=size, dtype=np.uint8) + 255

            image_container[:, (size[1] - w) // 2:(size[1] - w) // 2 + w] = image
        else:
            size = (int(w / (target_shape[1]/target_shape[0])), w)

            image_container = np.zeros(shape=size, dtype=np.uint8) + 255

            image_container[(size[0] - h) // 2:(size[0] - h) // 2 + h, :] = image

        return image_container

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here


        fname = dp['file_path']
        label = dp['InChI_text']
        InChI =dp['InChI']
        image_raw = cv2.imread(fname,-1)

        edge = 25
        image_raw = image_raw[edge:-edge, edge:-edge]

        if is_training:
            if random.uniform(0,1)>0.5:
                image_raw=self.addSaltNoise(image_raw,random.uniform(0.995,1))

        image_raw = self.letterbox(image_raw)

        ### make label
        label_padding = np.zeros(shape=[cfg.MODEL.train_length]) + self.word_tool.stoi['<pad>']
        label = self.word_tool.text_to_sequence(label)
        if len(label) > cfg.MODEL.train_length:
            label = label[:cfg.MODEL.train_length]

        label_padding[:len(label)] = label
        label_length=len(label)-1

        if is_training:


            transformed=self.train_trans(image=image_raw)

            image=transformed['image']
            image = np.expand_dims(image, 0)

            return 255 - image, label_padding, label_length


        else:
            transformed = self.val_trans(image=image_raw)

            image = transformed['image']
            image = np.expand_dims(image, 0)


            return 255 - image, label_padding, label_length, str(InChI)

