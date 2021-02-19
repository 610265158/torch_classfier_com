import pandas as pd
import os



reconstructed_data=pd.DataFrame(columns=['image_id','label'])


raw_data=pd.read_csv('~/Downloads/train/_annotations.csv')

imageid=raw_data['StudyInstanceUID'].values

imageid=[os.path.join(x,'.jpg') for x in imageid]

reconstructed_data['image_id']=imageid
reconstructed_data['label']=-1


all_label=['ETT - Abnormal',
           'ETT - Borderline',
           'ETT - Normal',
           'NGT - Abnormal',
           'NGT - Borderline',
           'NGT - Incompletely Imaged',
           'NGT - Normal',
           'CVC - Abnormal',
           'CVC - Borderline',
           'CVC - Normal',
           'Swan Ganz Catheter Present']


for k,label in enumerate(all_label):

    cur_index=(raw_data['label'].values==label)
    reconstructed_data['label'][cur_index]=k


reconstructed_data.to_csv('./train.csv',index=False)