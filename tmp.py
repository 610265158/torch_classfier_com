import cv2
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
data=pd.read_csv('../train/_annotations.csv')



fnames=data['StudyInstanceUID'].values


def get_label(data):

    data=data[2:-2].split('], [')

    res=[]

    for item in data:
        x,y=item.split(', ')
        res.append([int(x),int(y)])

    return np.array(res)



min_x=1
min_y=1
max_x=0
max_y=0
for i in tqdm(range(len(data))):

    cur_data=data.iloc[i]
    fname=cur_data['StudyInstanceUID']

    image_path= os.path.join('../train',fname+'.jpg')

    image=cv2.imread(image_path)
    h,w,c=image.shape
    if 'ETT' in cur_data['label']:
        res=get_label(cur_data['data'])

        min_x=min(min_x,np.min(res[:,0]/w))
        min_y = min(min_y,np.min(res[:, 1]/h))

        max_x = max(max_x, np.max(res[:, 0]/w))
        max_y = max(max_y, np.max(res[:, 1])/h)


        print(min_x,min_y,max_x,max_y)
        for i in range(res.shape[0]):
            cv2.circle(image,center=(res[i][0],res[i][1]),color=(0,0,255),radius=5)


        cv2.namedWindow('ss',0)
        cv2.imshow('ss',image)
        cv2.waitKey(0)



