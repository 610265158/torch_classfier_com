

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



class WordUtil():
    def __init__(self, df):
        words = set()
        for st in df['InChI']:
            words.update(set(st))
        len(words)

        vocab = list(words)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        self.stoi = {'C': 0, ')': 1, 'P': 2, 'l': 3, '=': 4, '3': 5, 'N': 6, 'I': 7, '2': 8, '6': 9, 'H': 10, '4': 11,
                'F': 12, '0': 13, '1': 14, '-': 15, 'O': 16, '8': 17,
                ',': 18, 'B': 19, '(': 20, '7': 21, 'r': 22, '/': 23, 'm': 24, 'c': 25, 's': 26, 'h': 27, 'i': 28,
                't': 29, 'T': 30, 'n': 31, '5': 32, '+': 33, 'b': 34, '9': 35,
                'D': 36, 'S': 37, '<sos>': 38, '<eos>': 39, '<pad>': 40}
        self.itos = {item[1]: item[0] for item in self.stoi.items()}


    def string_to_ints(self,string):

        l=[self.stoi['<sos>']]
        for s in string:
            l.append(self.stoi[s])
        l.append(self.stoi['<eos>'])
        return l
    def ints_to_string(self,l):
        return ''.join(list(map(lambda i:self.itos[i],l)))



class AlaskaDataIter():
    def __init__(self, df,
                 training_flag=True,shuffle=True):



        self.training_flag = training_flag
        self.shuffle = shuffle

        self.df=df

        self.word_tool=WordUtil(df)




        self.train_trans=A.Compose([A.RandomResizedCrop(height=cfg.MODEL.height,
                                                        width=cfg.MODEL.width,
                                                        scale=[0.9,1.]
                                                        ),
                                    A.HorizontalFlip(p=0.5),
                                    A.ShiftScaleRotate(p=0.5,
                                                       shift_limit=0.2,
                                                       scale_limit=0.2,
                                                       rotate_limit=20,
                                                       border_mode=cv2.BORDER_CONSTANT),

                                    A.RandomBrightnessContrast(brightness_limit=(0.2), contrast_limit=(0.2),
                                                             p=0.5),
                                    A.CLAHE(clip_limit=(1, 4), p=0.5),
                                    # A.OneOf([
                                    #     A.GridDistortion(num_steps=5, distort_limit=1.,border_mode=cv2.BORDER_CONSTANT),
                                    #     A.ElasticTransform(alpha=3,border_mode=cv2.BORDER_CONSTANT),
                                    # ], p=0.2),

                                    A.JpegCompression(p=0.2,
                                                      quality_lower=80,
                                                      quality_upper=100),
                                    A.OneOf([
                                        A.IAAAffine(mode='constant'),
                                        A.IAAPerspective(),
                                        A.IAAPiecewiseAffine(p=0.2),
                                        A.IAASharpen(p=0.2),
                                    ], p=0.2),

                                    # A.CoarseDropout(max_holes=6,max_width=64,max_height=64)
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
        label = dp['InChI']


        label_padding=np.zeros(shape=[cfg.MODEL.train_length])+self.word_tool.stoi['<pad>']
        try:

            image_raw = cv2.imread(fname,-1)


            if is_training:
                transformed=self.train_trans(image=image_raw)

                image=transformed['image']

            else:
                transformed = self.val_trans(image=image_raw)

                image = transformed['image']

            label=self.word_tool.string_to_ints(label)
            if len(label)>cfg.MODEL.train_length:
                label=label[:cfg.MODEL.train_length]
            label_padding[:len(label)]=label
        except:
            print(traceback.print_exc())
            logger.info('err happends with %s'% fname)
            image=np.zeros(shape=[cfg.MODEL.height,cfg.MODEL.width],dtype=np.uint8)
            label=np.zeros_like(label)

        image=np.stack([image,image,image],0)


        return image,label_padding
