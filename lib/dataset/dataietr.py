

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
                                           width=cfg.MODEL.width)



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

    def cutout(self,src,cover, max_pattern_ratio=0.05):
        cover_raw=np.array(cover)
        width_ratio = random.uniform(0, max_pattern_ratio)
        height_ratio = random.uniform(0, max_pattern_ratio)
        width = src.shape[1]
        height = src.shape[0]
        block_width = (width * width_ratio)//8*8
        block_height = (height * height_ratio)//8*8
        width_start = int(random.uniform(0, width - block_width))
        width_end = int(width_start + block_width)
        height_start = int(random.uniform(0, height - block_height))
        height_end = int(height_start + block_height)


        random_cover_pitch = cover[height_start:height_end, width_start:width_end, :]
        random_stego_pitch = src[height_start:height_end, width_start:width_end, :]

        src[height_start:height_end, width_start:width_end, :]= random_cover_pitch

        cover[height_start:height_end, width_start:width_end, :] = random_stego_pitch

        return src,cover,cover_raw

    def random_dash(self,src,how_many=8,block_size=64):

        ### we get the mask first

        def get_random_mask(image, block_size=32):

            h,w,c=image.shape
            mask = np.ones_like(image, dtype=np.uint8)
            for i in range(how_many):

                start_x = int(random.uniform(0,w-block_size))
                start_y = int(random.uniform(0, w - block_size))

                mask[start_y:start_y+block_size,start_x:start_x+block_size,:]=0
            return mask

        mask=get_random_mask(src,block_size)

        masked_src=src*mask


        return masked_src



    def random_resize(self,image,scale_range=[0.5,1]):

        scale=random.uniform(*scale_range)

        resize_method=random.choice([cv2.INTER_NEAREST,
                                     cv2.INTER_LINEAR,
                                     cv2.INTER_CUBIC,
                                     cv2.INTER_AREA])

        image_resized=cv2.resize(image,dsize=None,fx=scale,fy=scale,interpolation=resize_method)

        return image_resized


    def get_seg_label(self,image,extra_label, divide=4):

        h,w=image.shape
        seg_label=np.zeros([h//divide,w//divide,11])

        ann=0
        for i in range(len(extra_label)):
            one_tap=np.zeros([h//divide,w//divide])

            points=np.array(extra_label[i]['transkps'])//divide
            label=extra_label[i]['label']
            if label!=-1:
                ann=1
                for j in range(points.shape[0]-1):
                    cv2.line(one_tap,pt1=(int(points[j][0]),int(points[j][1])),
                             pt2=(int(points[j+1][0]),int(points[j+1][1])),
                             color=(255))

                one_tap = cv2.blur(one_tap, ksize=(5, 5))
                one_tap[one_tap>0]=1
                seg_label[:,:,label]=one_tap


        return seg_label,ann

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here


        fname = dp['file_path']
        label = dp['InChI_text']





        image_raw = cv2.imread(fname,-1)

        if is_training:

            label_padding = np.zeros(shape=[cfg.MODEL.train_length]) + self.word_tool.stoi['<pad>']
            transformed=self.train_trans(image=image_raw)

            image=transformed['image']
            label = self.word_tool.text_to_sequence(label)
            if len(label) > cfg.MODEL.train_length:
                label = label[:cfg.MODEL.train_length]

            label_padding[:len(label)] = label

            image = np.stack([image, image, image], 0)

            return image, label_padding
        else:
            transformed = self.val_trans(image=image_raw)

            image = transformed['image']
            image = np.stack([image, image, image], 0)
            return image

