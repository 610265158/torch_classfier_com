
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import AlaskaDataIter
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold

from train_config import config as cfg

import setproctitle


setproctitle.setproctitle("comp")


def main():
    n_fold=5

    data=pd.read_csv(cfg.DATA.data_file)

    def get_train_file_path(image_id):
        return cfg.DATA.data_root_path+"/{}/{}/{}/{}.png".format(
            image_id[0], image_id[1], image_id[2], image_id
        )


    data['file_path'] = data['image_id'].apply(get_train_file_path)

    n_fold = 5

    def split(data,n_fold=5):


        data['fold'] = -1
        Fold = KFold(n_splits=n_fold, shuffle=True, random_state=cfg.SEED)

        for fold, (train_index, test_index) in enumerate(Fold.split(data)):
            data['fold'][test_index] = fold

        return data

    data = split(data,n_fold)



    for fold in range(n_fold):
        ###build dataset

        train_ind = data[data['fold'] != fold].index.values
        train_data = data.iloc[train_ind].copy()
        val_ind = data[data['fold'] == fold].index.values
        val_data = data.iloc[val_ind].copy()


        trainds=AlaskaDataIter(train_data,training_flag=True,shuffle=False)
        train_ds = DataLoader(trainds,
                              cfg.TRAIN.batch_size,
                              num_workers=cfg.TRAIN.process_num,
                              shuffle=True)

        valds = AlaskaDataIter(val_data,training_flag=False,shuffle=False)
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


                for i in range(images.shape[0]):


                    example_image=np.array(images[i],dtype=np.uint8)
                    example_image=np.transpose(example_image,[1,2,0])
                    example_label=np.array(labels[i])

                    print(example_label)
                    print(example_label.shape)
                    cv2.imshow('example',example_image)

                    cv2.waitKey(0)

        ### train
        trainer.custom_loop()

if __name__=='__main__':
    main()