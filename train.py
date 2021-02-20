
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import AlaskaDataIter
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train_config import config as cfg

import setproctitle


setproctitle.setproctitle("comp")


def main():
    n_fold=5

    data=pd.read_csv(cfg.DATA.data_file)

    for fold in range(n_fold):
        ###build dataset

        train_ind = data[data['fold'] != fold].index.values
        train_data = data.iloc[train_ind].copy()
        val_ind = data[data['fold'] == fold].index.values
        val_data = data.iloc[val_ind].copy()


        trainds=AlaskaDataIter(train_data)
        train_ds = DataLoader(trainds,
                              cfg.TRAIN.batch_size,
                              num_workers=cfg.TRAIN.process_num,
                              shuffle=True)

        valds = AlaskaDataIter(val_data)
        test_ds = DataLoader(valds,
                             cfg.TRAIN.batch_size,
                             num_workers=cfg.TRAIN.process_num,
                             shuffle=False)

        ###build trainer
        trainer = Train(train_ds=train_ds,val_ds=test_ds,fold=fold)

        print('it is here')
        if cfg.TRAIN.vis:
            print('show it, here')
            for images, labels in train_ds:


                # images, mask, labels = cutmix_numpy(images, mask, labels, 0.5)
                print(images.shape)
                for i in range(images.shape[0]):
                    example_image=np.array(images[i],dtype=np.uint8)
                    example_image=np.transpose(example_image,[1,2,0])
                    example_label=np.array(labels[i])

                    _h, _w, _ = example_image.shape

                    print(example_label)

                    cv2.imshow('example',example_image)

                    cv2.waitKey(0)

        ### train
        trainer.custom_loop()

if __name__=='__main__':
    main()