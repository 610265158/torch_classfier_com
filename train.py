import torch

from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import AlaskaDataIter
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold

from train_config import config as cfg
from make_data import Tokenizer
import setproctitle


setproctitle.setproctitle("comp")


def main():

    def get_folds(n_fold = 5):
        data = pd.read_csv('../train2.csv')

        def get_train_file_path(image_id):
            return cfg.DATA.data_root_path+"/{}/{}/{}/{}.png".format(
                image_id[0], image_id[1], image_id[2], image_id
            )


        data['file_path'] = data['image_id'].apply(get_train_file_path)



        def split(train, n_fold=5):


            folds = train.copy()
            Fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cfg.SEED)
            for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
                folds.loc[val_index, 'fold'] = int(n)
            folds['fold'] = folds['fold'].astype(int)
            print(folds.groupby(['fold']).size())




            return  folds


        data = split(data,n_fold)
        return data
    n_fold = 5
    data=get_folds()

    def get_token():
        token_tools=Tokenizer()
        token_tools.stoi=np.load("../tokenizer.stio.npy", allow_pickle=True).item()
        token_tools.itos = np.load("../tokenizer.itos.npy", allow_pickle=True).item()

        return token_tools

    token_tools=get_token()

    for fold in range(n_fold):
        ###build dataset

        train_ind = data[data['fold'] != fold].index.values
        train_data = data.iloc[train_ind].copy()
        val_ind = data[data['fold'] == fold].index.values
        val_data = data.iloc[val_ind].copy()




        ###build trainer
        trainer = Train(train_df=train_data,
                        val_df=val_data,
                        fold=fold,
                        tokenizer=token_tools)

        print('it is here')
        if cfg.TRAIN.vis:
            print('show it, here')

            trainds = AlaskaDataIter(train_data, token_tools, training_flag=True, shuffle=False)
            train_ds = DataLoader(trainds,
                                  cfg.TRAIN.batch_size,
                                  num_workers=cfg.TRAIN.process_num,
                                  shuffle=True)

            valds = AlaskaDataIter(val_data, token_tools, training_flag=False, shuffle=False)
            test_ds = DataLoader(valds,
                                 cfg.TRAIN.batch_size,
                                 num_workers=cfg.TRAIN.process_num,
                                 shuffle=False)
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