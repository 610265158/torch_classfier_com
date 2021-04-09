import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from train_config import config as cfg
# def get_test_df():
#     data = pd.read_csv('../train2.csv')
#
#     def get_train_file_path(image_id):
#         return cfg.DATA.data_root_path + "/{}/{}/{}/{}.png".format(
#             image_id[0], image_id[1], image_id[2], image_id
#         )
#
#     data['file_path'] = data['image_id'].apply(get_train_file_path)
#     return data
#
#
#
# ds=get_test_df()
#
# print(ds.head())
#
# fnames=ds['file_path']
#
# H=[]
# W=[]
# for fn in tqdm(fnames):
#     image=cv2.imread(fn,-1)
#
#     h,w,=image.shape
#     H.append(h)
#     W.append(w)
#
# ds['height']=H
# ds['width']=W
# ds.to_csv('./detail.csv',index=False)



ds=pd.read_csv('../detail.csv')
H=ds['height']
W=ds['width']

w_h=W/H


print('mean',np.mean(w_h))
print('max',np.max(w_h))
print('min',np.min(w_h))
print('median',np.median(w_h))
print('h/w> 1.5',np.sum(w_h>2))
print('h/w> 2',np.sum(w_h>2))
print('h/w< 3',np.sum(w_h<3))
###draw
age1=[]
age2=[]
age3=[]
age4=[]
age5=[]
for i in W:
    if 0<=i<200:
        age1.append(i)
    elif 200<=i<400:
        age2.append(i)
    elif 400<=i<600:
        age3.append(i)
    elif 600<=i<800:
        age4.append(i)
    else:
        age5.append(i)
print(len(age1))
print(len(age2))
print(len(age3))
print(len(age4))
print(len(age5))
index=['100~200','200~400','400~600','600~800','>800']
values=[len(age1),len(age2),len(age3),len(age4),len(age5)]
plt.bar(index,values)
plt.show()