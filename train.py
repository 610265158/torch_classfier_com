
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
    def get_fold(n_fold=n_fold):

        def get_train_file_path(image_id):
            return cfg.DATA.data_root_path + "/{}/{}.npy".format(
                image_id[0], image_id
            )

        data=pd.read_csv(cfg.DATA.data_file)

        data['file_path'] = data['id'].apply(get_train_file_path)

        def filter(image_id):
            return '/0/' in image_id

        choose_index=data['file_path'].apply(filter)
        data=data.loc[choose_index]

        folds = data.copy()
        Fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cfg.SEED)
        for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['target'])):
            folds.loc[val_index, 'fold'] = int(n)
        folds['fold'] = folds['fold'].astype(int)
        return folds


    data=get_fold(n_fold)

    for fold in range(n_fold):
        ###build dataset

        train_ind = data[data['fold'] != fold].index.values
        train_data = data.iloc[train_ind].copy()
        val_ind = data[data['fold'] == fold].index.values
        val_data = data.iloc[val_ind].copy()

        ###build trainer

        if cfg.TRAIN.vis:
            print('show it, here')

            trainds = AlaskaDataIter(train_data, training_flag=True, shuffle=False)
            train_ds = DataLoader(trainds,
                                  cfg.TRAIN.batch_size,
                                  num_workers=cfg.TRAIN.process_num,
                                  shuffle=True)

            valds = AlaskaDataIter(val_data, training_flag=False, shuffle=False)
            test_ds = DataLoader(valds,
                                 cfg.TRAIN.batch_size,
                                 num_workers=cfg.TRAIN.process_num,
                                 shuffle=False)
            for images, labels in train_ds:

                # images, mask, labels = cutmix_numpy(images, mask, labels, 0.5)

                for i in range(images.shape[0]):
                    example_image = np.array(images[i], dtype=np.uint8)
                    example_image = np.transpose(example_image, [1, 2, 0])
                    example_label = np.array(labels[i])

                    print(example_label)
                    print(example_label.shape)
                    cv2.imshow('example', example_image)

                    cv2.waitKey(0)

        ###build trainer
        trainer = Train(train_df=train_data,
                        val_df=val_data,
                        fold=fold)

        ### train
        trainer.custom_loop()

if __name__=='__main__':
    main()