import os
import shutil

from lib.dataset.dataietr import DataIter, data_info

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from lib.helper.logger import logger
from train_config import config as cfg

from sklearn.metrics import confusion_matrix

import setproctitle

from train_config import config as cfg
import albumentations as A
setproctitle.setproctitle("eval")
import torch
from tqdm import tqdm


weight='./models/fold1_epoch_23_val_loss0.315470.pth'
fold=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class AlaskaDataIter():
    def __init__(self, data ):



        self.raw_data_set_size = None     ##decided by self.parse_file
        self.lst = self.parse_file(data)

        self.val_trans=A.Compose([ #A.CenterCrop(height=cfg.MODEL.height,
                                   #             width=cfg.MODEL.width, p=1.),

                                   A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width)

                                  ])
    def __getitem__(self, item):

        return self.single_map_func(self.lst[item])

    def __len__(self):
        assert self.raw_data_set_size is not None

        return self.raw_data_set_size

    def parse_file(self,df):
        '''
        :return:
        '''

        ann_info = data_info(df)
        all_samples = ann_info.get_all_sample()
        self.raw_data_set_size=len(all_samples)

        return all_samples


    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.val_trans(image=image)['image']

        #         image=cv2.resize(image,(512,512))

        #image_transpose = image.transpose(1, 0, 2)
        #         image_90=np.rot90(image,1)
        #         image_180 = np.rot90(image, 2)
        #         image_270 = np.rot90(image, 3)

        image_fliplr = np.fliplr(image)
        #         image_fliplr_90 = np.rot90(image, 1)
        #         image_fliplr_180 = np.rot90(image, 2)
        #         image_fliplr_270 = np.rot90(image, 3)

        image_flipud = np.flipud(image)

        image_batch = np.stack([image,
                                #image_transpose,
                                image_fliplr,
                                image_flipud])

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        return image, image_batch
    def single_map_func(self, dp,):
        """Data augmentation function."""
        ####customed here

        fname = os.path.join(cfg.DATA.data_root_path,dp[0])

        label = int(dp[1])

        image = cv2.imread(fname, -1)


        _,image_batch=self.preprocess_func(image)


        return fname,image_batch,label

class ACCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0, 1])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()



        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))

    @property
    def avg(self):
        right = (self.y_pred == self.y_true).astype(np.float)

        ###

        for i in range(5):
            index = (self.y_true == i)

            cur_y_true = self.y_true[index]
            cur_y_pre = self.y_pred[index]

            cur_acc = np.sum(cur_y_true == cur_y_pre) / np.sum(index)

            logger.info(' for class %d, acc %.6f' % (i, cur_acc))

        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1, 2,3,4])
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
        cm = np.around(cm,decimals=2)
        print(cm)

        # plot_confusion_matrix(cm, [0,1,2,3,4], "HAR Confusion Matrix")
        # print('xxx')
        # plt.savefig('HAR_cm.png', format='png')
        # plt.show()


        return np.sum(right) / self.y_true.shape[0]
from lib.core.base_trainer.model import Net




model=Net().to(device)


model.load_state_dict(torch.load(weight, map_location=device))
### load your weights
model.eval()

def main():


    ### 5 fold

    def split(n_fold=5):

        data=pd.read_csv(cfg.DATA.data_file)

        data['fold'] = -1
        Fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cfg.SEED)

        if "merge" in cfg.DATA.data_file:
            data_2020=data[data['source']==2020]
            data_2019=data[data['source']==2019]
            for fold, (train_index, test_index) in enumerate(Fold.split(data_2020, data_2020['label'])):
                data_2020['fold'][test_index] = fold


            data=data_2020.append(data_2019)
        else:

            for fold, (train_index, test_index) in enumerate(Fold.split(data, data['label'])):
                data['fold'][test_index] = fold



        return data

    n_fold = 5
    data=split(n_fold)


    ###build dataset

    train_ind = data[data['fold'] != fold].index.values
    train_data = data.iloc[train_ind].copy()
    val_ind = data[data['fold'] == fold].index.values
    val_data = data.iloc[val_ind].copy()

    DATASET=AlaskaDataIter(train_data)
    ###build modeler

    acc_score = ACCMeter()



    for i in tqdm(range(len(DATASET))):


        fname,float_image, target= DATASET.__getitem__(i)


        input = torch.from_numpy(float_image).to(device).float()
        target = torch.tensor(target).to(device).long()

        with torch.no_grad():
            output = model(input)

            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.cpu().numpy()
            output = np.mean(output, axis=0)
            pre_label = np.argmax(output)
            acc_score.update(target, pre_label)
            #
            # if pre_label!=target:
            #     # iimg=float_image[0,...]
            #     # iimg=np.transpose(iimg,[1,2,0]).astype(np.uint8)
            #     #
            #     #
            #     # print('fname',fname, 'predict',pre_label,  'target ' ,target)
            #     # iimg = cv2.cvtColor(iimg, cv2.COLOR_BGR2RGB)
            #     # cv2.imshow('xx',iimg)
            #     # kk=cv2.waitKey(0)
            #     #
            #     # if kk==115:
            #     #     shutil.copy(fname,os.path.join('./err_labeled',fname.split('/')[-1]))
            #
            #     shutil.copy(fname, os.path.join('./err_labeled', fname.split('/')[-1]))
                # iimg=


    print('cur fold :%d acc:%5.f',fold, acc_score.avg)





if __name__=='__main__':
    main()