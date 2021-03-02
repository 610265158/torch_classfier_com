from lib.core.base_trainer.metric import ROCAUCMeter
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

from lib.core.base_trainer.model import Net
import torch
from tqdm import tqdm
weights=['./models/fold0_epoch_8_val_loss_0.141222_rocauc_0.952141.pth',
         './models/fold1_epoch_9_val_loss_0.138928_rocauc_0.953727.pth',
         './models/fold2_epoch_8_val_loss_0.133868_rocauc_0.958444.pth',
         './models/fold3_epoch_8_val_loss_0.138063_rocauc_0.955710.pth',
         './models/fold4_epoch_8_val_loss_0.137824_rocauc_0.952578.pth']

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=Net().to(device)
def main():
    n_fold=5


    rocauc_score = ROCAUCMeter()
    for fold in range(n_fold):
        print(' infer with fold %d' % (fold))
        model.load_state_dict(torch.load(weights[fold], map_location=device))

        model.eval()


        data = pd.read_csv(cfg.DATA.data_file)
        val_ind = data[data['fold'] == fold].index.values
        val_data = data.iloc[val_ind].copy()

        valds = AlaskaDataIter(val_data,training_flag=False,shuffle=False)
        test_ds = DataLoader(valds,
                             cfg.TRAIN.batch_size,
                             num_workers=cfg.TRAIN.process_num,
                             shuffle=False)



        with torch.no_grad():
            for  (images, target) in tqdm(test_ds):
                data = images.to(device).float()
                target = target.to(device).long()

                output = model(data)



                rocauc_score.update(target, output)


    print('5 fold roc aud score is ,' ,rocauc_score.avg)

if __name__=='__main__':
    main()