#-*-coding:utf-8-*-

import os

import random
import cv2
import pandas as pd

ratio=0.8
data_set_dir='./IMAGENET'

checker_image_state=False



def prepare_data():
    train_df = pd.DataFrame(columns=('fname', 'class'))
    labels=os.listdir(data_set_dir)

    ##filter
    labels=["人像","古建筑","夜色","家居","日出日落",\
                   "植物","海天淡蓝色","美食","街景","animal",\
                   "airplane","bicycle","boat","bus","car",\
                   "motorcycle","train","truck"
                   "background"]


    syntext_f=open('label.txt','w')
    for label in labels:
        if label=='background':
            message = label + ' ' + str(-1) + '\n'
        else:
            message = label + ' ' +str(labels.index(label)) +'\n'
        syntext_f.write(message)


    for label in labels:
        cur_dir=os.path.join(data_set_dir,label)

        pic_list=os.listdir(cur_dir)

        random.shuffle(pic_list)

        num_data=len(pic_list)


        for pic in pic_list:
            cur_path=os.path.join(cur_dir,pic)

            try:
                if checker_image_state:
                    img=cv2.imread(cur_path)
                else:
                    img='ok'
                if img is not  None:
                    if label == 'background':
                        tmp_item=pd.Series({'fname': cur_path, 'class': -1})
                    else:
                        tmp_item = pd.Series({'fname': cur_path, 'class': labels.index(label)})
                    train_df=train_df.append(tmp_item,ignore_index=True)

            except:
                continue


    train_df.to_csv('train.csv',index=False)


    syntext_f.close()


if __name__=='__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", required=False, default=0.8,type=float, help="train val split ratio")
    ap.add_argument("--data_dir", required=False, default="IMAGENET", help="train val split ratio")

    args = ap.parse_args()

    ratio = args.ratio
    data_set_dir=args.data_dir


    prepare_data()