

import random
import cv2
import json
import numpy as np
import copy
import matplotlib.pyplot as plt
from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator, BatchData, MultiProcessPrefetchData, PrefetchDataZMQ, RepeatedData
import time


from lib.dataset.augmentor.augmentation import Rotate_aug,\
                                                Affine_aug,\
                                                Mirror,\
                                                Padding_aug,\
                                                Img_dropout,Random_crop



from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter

from train_config import config as cfg
import albumentations as A
import os

class data_info(object):
    def __init__(self,df,training=True):
        self.df=df
        self.metas=[]
        self.training=training

        self.load_anns()
    def one_hot(self,p,length):
        label=np.zeros(shape=length)
        label[p]=1
        return label

    def load_anns(self):

        num_sample=len(self.df)

        for i in range(num_sample):

            cur_line= self.df.iloc[i]

            fname = cur_line['image_id']
            label = cur_line['label']





            self.metas.append([fname, label])

            ###some change can be made here

        logger.info('the datasets contains %d samples'%(num_sample))
        logger.info('the datasets contains %d samples after filter' % (num_sample))

    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas
#
#
class DataIter():
    def __init__(self,data,training_flag=True,shuffle=True):

        self.shuffle=shuffle
        self.training_flag=training_flag
        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size


        if not training_flag:
            self.process_num=1
        self.generator = AlaskaDataIter(data, self.training_flag,self.shuffle)

        self.ds=self.build_iter()

        self.size = self.__len__()


    def parse_file(self,im_root_path,ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")


    def build_iter(self):

        ds = DataFromGenerator(self.generator)
        ds = RepeatedData(ds, -1)
        ds = BatchData(ds, self.batch_size)
        if not cfg.TRAIN.vis:
            ds = PrefetchDataZMQ(ds, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    def __iter__(self):

        for i in range(self.size):
            one_batch = next(self.ds)

            return one_batch[0], one_batch[1], one_batch[2]

    def __call__(self, *args, **kwargs):


        for i in range(self.size):


            one_batch=next(self.ds)

            data,label=one_batch[0],one_batch[1]

            return data,label



    def __len__(self):
        return len(self.generator)//self.batch_size

    def _map_func(self,dp,is_training):

        raise NotImplementedError("you need implemented the map func for your data")




class AlaskaDataIter():
    def __init__(self, data, training_flag=True,shuffle=True):



        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None     ##decided by self.parse_file


        self.lst = self.parse_file(data)




        self.train_trans=A.Compose([A.RandomResizedCrop(scale=(0.7,1.3),
                                                        height=cfg.MODEL.height,
                                                        width=cfg.MODEL.width
                                                        ),
                                    A.ShiftScaleRotate(shift_limit=0.0625,
                                                 scale_limit=0.50,
                                                 rotate_limit=180,
                                                 p=1.,
                                                 border_mode=cv2.BORDER_CONSTANT,
                                                 value=0,
                                                 interpolation=cv2.INTER_AREA),
                                    A.RandomBrightnessContrast(p=0.75,brightness_limit=0.1,contrast_limit=0.2),

                                    A.CLAHE(clip_limit=4.0, p=0.7),
                                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                                                  val_shift_limit=10, p=0.5),

                                    A.OneOf([
                                        A.MotionBlur(blur_limit=5),
                                        A.MedianBlur(blur_limit=5),
                                        A.GaussianBlur(blur_limit=5),
                                        A.GaussNoise(var_limit=(5.0, 30.0)),
                                    ], p=0.7),

                                    A.Resize(height=cfg.MODEL.height,
                                             width=cfg.MODEL.width),
                              ])


        self.val_trans=A.Compose([A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width)

                                  ])
    def __call__(self, *args, **kwargs):
        idxs = np.arange(len(self.lst))

        # while True:
        if self.shuffle:
            np.random.shuffle(idxs)
        for k in idxs:
            yield self.single_map_func(self.lst[k], self.training_flag)

    def __len__(self):
        assert self.raw_data_set_size is not None

        return self.raw_data_set_size


    def parse_file(self,df):
        '''
        :return:
        '''
        logger.info("[x] Get dataset from csv")

        ann_info = data_info(df)
        all_samples = ann_info.get_all_sample()
        self.raw_data_set_size=len(all_samples)

        return all_samples

    def cracy_rotate(self, image, block_nums=2):

        block_size = 512 // block_nums
        for i in range(block_nums):
            for j in range(block_nums):
                cur_pitch = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, :]

                if random.uniform(0, 1) >= 0.5:
                    cur_pitch = np.fliplr(cur_pitch)
                if random.uniform(0, 1) >= 0.5:
                    cur_pitch = np.flipud(cur_pitch)

                if random.uniform(0, 1) >= 0.5:
                    random_angle = [-1, 1]
                    cur_pitch = np.rot90(cur_pitch, random.choice(random_angle))
                image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, :] = cur_pitch

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




    def onehot(self,lable,depth=1000):
        one_hot_label=np.zeros(shape=depth)

        if lable!=-1:
            one_hot_label[lable]=1

        return one_hot_label

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here

        fname = os.path.join(cfg.DATA.data_root_path,dp[0])

        label = int(dp[1])

        image = cv2.imread(fname, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if is_training:

            image=self.train_trans(image=image)['image']

            ###cutout
            if random.uniform(0, 1) >= 0.5:
                image=self.random_dash(image,8,32)
            if random.uniform(0, 1) >= 0.5:
                image,_= Mirror(image)

        else:

            image=self.val_trans(image=image)['image']


        image = np.transpose(image, axes=[2, 0, 1])

        image=image.astype(np.uint8)


        return image,label
