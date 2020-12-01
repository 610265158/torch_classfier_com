import torch

import os
import cv2
import numpy as np

from tqdm import tqdm

import pandas as pd
import albumentations as A

from train_config import config as cfg


from lib.core.base_trainer.model import Net


data_dir='./tmp'

weights=['./fold0_epoch_49_val_loss0.124049.pth']

label_file='label.txt'



DEBUG=True
def get_label_map():
    txt_file=open(label_file,'r')
    lines=txt_file.readlines()

    label_map={}
    for line in lines:
        line=line.rstrip()

        label,id=line.split(' ')
        label_map[int(id)]=label

    return label_map



label_map=get_label_map()

class DatasetTest:
    def __init__(self, test_data_dir):
        self.ds = self.get_list(test_data_dir)

        self.root_dir = test_data_dir

        self.val_trans = A.Compose([A.Resize(height=cfg.MODEL.height,
                                        width=cfg.MODEL.width),
                               A.CenterCrop(height=cfg.MODEL.height, width=cfg.MODEL.width),
                               ])
    def get_list(self, dir):
        pic_list = os.listdir(dir)

        return pic_list


    def __len__(self):
        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = self.val_trans(image=image)['image']


        flip_ed_image=np.fliplr(image)
        image_batch = np.stack([image,flip_ed_image])

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        # image = np.expand_dims(image, 0)
        return image_batch

    def __getitem__(self, item):
        fname = self.ds[item]

        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, -1)
        image = self.preprocess_func(image)

        return fname, image





model=Net()


dataset=DatasetTest(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result = []

for weight in weights:

    res={'fname':[],'pred':[],'score':[]}

    model.load_state_dict(torch.load(weight, map_location=device))
    model.eval()
    model.to(device)
    ds = DatasetTest(test_data_dir=data_dir)

    for i in tqdm(range(len(ds))):

        try:
            fname, img = ds.__getitem__(i)
        except:
            continue
        res['fname'].append(fname)

        img_show=np.transpose(img[0],axes=[1,2,0]).astype(np.uint8)

        img = torch.from_numpy(img)
        img = img.to(device)
        with torch.no_grad():
            y_pred = model(img)


        y_pre_act = torch.sigmoid(y_pred).data.cpu().numpy()
        y_pre_act = np.mean(y_pre_act,axis=0)


        y_pre_label=int(np.argmax(y_pre_act))

        score=y_pre_act[y_pre_label]
        res['pred'].append(y_pre_label)
        res['score'].append(score)
        if DEBUG:

            print(fname, 'blones to ', label_map[y_pre_label], score)
            cv2.namedWindow('example',0)
            cv2.imshow('example',img_show)
            cv2.waitKey(0)

    result.append(res)

#####  enmsemble


####
submission = pd.DataFrame(result[0])
submission.to_csv('pseudo_label.csv', index=False)
submission.head()