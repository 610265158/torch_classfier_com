import cv2
import os
from tqdm import tqdm
import pandas as pd
from train_config import config as cfg
data_dir='./err_labeled'



label_data=data=pd.read_csv(cfg.DATA.data_file)

image_ids=label_data['image_id']


__labels=label_data['label']

print((__labels==0).sum())


for id in image_ids:


    image_path = os.path.join(cfg.DATA.data_root_path, id)


    img=cv2.imread(image_path)

    curline=label_data[label_data['image_id']==id]
    gt=curline['label'].values



    if gt==4 and (pd.isna(curline['fixed'].values[0])):

        print('gt is ',gt)
        print(id)
        cv2.imshow('ss',img)
        cv2.waitKey(0)






#
#
# image_list=os.listdir(data_dir)
#
#
# for pic in tqdm(image_list):
#     image_path=os.path.join(data_dir,pic)
#
#
#     img=cv2.imread(image_path)
#
#     gt=label_data[label_data['image_id']==pic]['label'].values
#
#
#     print('gt is ',gt)
#     cv2.imshow('ss',img)
#     cv2.waitKey(0)
