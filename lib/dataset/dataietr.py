

import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
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

class data_info(object):
    def __init__(self,df,training=True):
        self.df=df
        self.metas=[]
        self.training=training
        self.ann = pd.read_csv('../train/_annotations.csv')


        self.load_anns()



    def load_anns(self):
        def get_label(data):

            data = data[2:-2].split('], [')

            res = []

            for item in data:
                x, y = item.split(', ')
                res.append([int(x), int(y)])

            return np.array(res)

        num_sample=len(self.df)


        target_cols=['ETT - Abnormal','ETT - Borderline','ETT - Normal',
                     'NGT - Abnormal','NGT - Borderline','NGT - Incompletely Imaged','NGT - Normal',
                     'CVC - Abnormal','CVC - Borderline','CVC - Normal','Swan Ganz Catheter Present']



        max_ann=0
        for i in range(num_sample):

            cur_line= self.df.iloc[i]



            fname = str(cur_line['StudyInstanceUID']).rstrip()

            extra_label = []
            if fname in self.ann['StudyInstanceUID'].values:

                cur_data=self.ann[self.ann['StudyInstanceUID']==fname]



                for k in range(len(cur_data)):
                    cur_label={}
                    cur_label['label']=target_cols.index(cur_data.iloc[k]['label'])


                    points=get_label(cur_data.iloc[k]['data'])
                    points[points<0]=0
                    cur_label['keypoints']=points
                    cur_label['num_points'] = len(points)
                    extra_label.append(cur_label)




                # ann_label=cur_data['label']
                # keypoint=get_label(cur_data['data'].values)

                # print(keypoint)

            if len(extra_label)>max_ann:
                max_ann=len(extra_label)
            label = cur_line[target_cols].values


            self.metas.append([fname, label,extra_label])

            ###some change can be made here

        logger.info('the datasets contains %d samples'%(num_sample))
        logger.info('the datasets contains %d samples after filter' % (num_sample))

    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas
#
#



class AlaskaDataIter():
    def __init__(self, data,
                 training_flag=True,shuffle=True):



        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None     ##decided by self.parse_file


        self.lst = self.parse_file(data)

        #self.data_distribution=self.balance_data(self.lst)

        #
        #
        # self.lst =[]
        # for k,v in self.data_distribution.items():
        #     self.lst+=v
        #
        # logger.info('after balance contains%d samples'%len(self.lst))
        self.train_trans=A.Compose([A.RandomResizedCrop(height=cfg.MODEL.height,
                                                        width=cfg.MODEL.width,
                                                        scale=[0.9,1.]
                                                        ),
                                    #A.HorizontalFlip(p=0.5),
                                    A.ShiftScaleRotate(p=0.7,
                                                       shift_limit=0.2,
                                                       scale_limit=0.2,
                                                       rotate_limit=20,
                                                       border_mode=cv2.BORDER_CONSTANT),

                                    A.RandomBrightnessContrast(brightness_limit=(0.2), contrast_limit=(0.2),
                                                             p=0.7),
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
                                         keypoint_params=A.KeypointParams(format='xy'),
                                         additional_targets={'keypoints1': 'keypoints',
                                                             'keypoints2': 'keypoints',
                                                             'keypoints3': 'keypoints',
                                                             'keypoints4': 'keypoints',
                                                             'keypoints5': 'keypoints',}
                              )


        self.val_trans=A.Compose([

                                   A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width)

                                  ])
    # def __call__(self, *args, **kwargs):
    #
    #     idxs = np.arange(len(self.lst))
    #
    #     # while True:
    #     if self.shuffle:
    #         np.random.shuffle(idxs)
    #     for k in idxs:
    #         yield self.single_map_func(self.lst[k], self.training_flag)

    def __getitem__(self, item):

        return self.single_map_func(self.lst[item], self.training_flag)

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

                one_tap = cv2.blur(one_tap, ksize=(7, 7))
                one_tap[one_tap>0]=1
                seg_label[:,:,label]=one_tap


        return seg_label,ann

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here

        fname = os.path.join(cfg.DATA.data_root_path,dp[0]+'.jpg')

        label = np.array(dp[1])
        extra_label=dp[2]



        for i in range(6-len(extra_label)):
            tmp={'keypoints':[],
                 'label':-1}

            extra_label.append(tmp)





        try:

            image_raw = cv2.imread(fname,-1)
            h,w=image_raw.shape
            for i in range(len(extra_label)):
                pp= extra_label[i]["keypoints"]

                if len(pp)>0:
                    pp[:,0][pp[:,0]>=w]=w-1
                    pp[:, 1][pp[:, 1] >= h] = h - 1
            if is_training:
                transformed=self.train_trans(image=image_raw,
                                             keypoints =extra_label[0]['keypoints'],
                                             keypoints1=extra_label[1]['keypoints'],
                                             keypoints2=extra_label[2]['keypoints'],
                                             keypoints3=extra_label[3]['keypoints'],
                                             keypoints4=extra_label[4]['keypoints'],
                                             keypoints5=extra_label[5]['keypoints'],
                                             )

                image=transformed['image']
                extra_label[0]['transkps']=transformed['keypoints']
                extra_label[1]['transkps'] = transformed['keypoints1']
                extra_label[2]['transkps'] = transformed['keypoints2']
                extra_label[3]['transkps'] = transformed['keypoints3']
                extra_label[4]['transkps'] = transformed['keypoints4']
                extra_label[5]['transkps'] = transformed['keypoints5']

                mask,mask_weight=self.get_seg_label(image,extra_label)

                # for i in range(len(extra_label)):
                #
                #     kps=np.array(extra_label[i]['transkps'],dtype=np.int)
                #
                #     if len(kps)>0:
                #         for i in range(kps.shape[0]):
                #             cv2.circle(image, center=(int(kps[i][0]), int(kps[i][1])), color=(0, 0, 255), radius=5,thickness=5)







            else:
                transformed = self.val_trans(image=image_raw)

                image = transformed['image']
                image = np.expand_dims(image, axis=0)

                image = np.concatenate([image, image, image], axis=0)
                label = np.array(dp[1], dtype=np.int)
                return image,label


        except:
            print(traceback.print_exc())
            logger.info('err happends with %s'% fname)
            image=np.zeros(shape=[cfg.MODEL.height,cfg.MODEL.width],dtype=np.uint8)
            label=np.zeros_like(label)

            mask = np.zeros([cfg.MODEL.height//4,cfg.MODEL.width//4,11])
            mask_weight=0
        image = np.expand_dims(image,axis=0)

        image = np.concatenate([image,image,image],axis=0)

        label = np.array(dp[1],dtype=np.int)

        mask=np.transpose(mask,axes=[2,0,1])

        return image,label,mask,mask_weight
