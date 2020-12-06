
import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import os
import timm
import cv2
import numpy as np
import pandas as pd
def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-5):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # self.mean_tensor=torch.from_numpy(cfg.DATA.PIXEL_MEAN ).float().cuda()
        # self.std_val_tensor = torch.from_numpy(cfg.DATA.PIXEL_STD).float().cuda()
        # self.model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        # self.model = timm.create_model('mobilenetv2_110d', pretrained=True)

        # self.model = timm.create_model('mobilenetv2_110d', pretrained=True)
        self.model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.dropout=nn.Dropout(0.5)

        self._fc = nn.Linear(1536 , num_classes, bias=True)

    def forward(self, inputs):

        #do preprocess

        input_iid = inputs
        input_iid=input_iid/255.
        bs = input_iid.size(0)
        # Convolution layers
        x = self.model.forward_features(input_iid)
        fm = self._avg_pooling(x)
        fm = fm.view(bs, -1)
        feature=self.dropout(fm)


        x = self._fc(feature)


        return x




class DatasetTest():
    def __init__(self, test_data_dir):
        self.ds = self.get_list(test_data_dir)

        self.root_dir = test_data_dir

    def get_list(self, dir):
        pic_list = os.listdir(dir)

        return pic_list

    def __len__(self):

        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(512,512))

        image_90=np.rot90(image,1)
        image_180 = np.rot90(image, 2)
        image_270 = np.rot90(image, 3)

        image_fliplr = np.fliplr(image)
        image_fliplr_90 = np.rot90(image, 1)
        image_fliplr_180 = np.rot90(image, 2)
        image_fliplr_270 = np.rot90(image, 3)

        image_batch = np.stack([image,image_90,image_180,image_270,
                                image_fliplr,image_fliplr_90,image_fliplr_180,image_fliplr_270])

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        return image, image_batch

    def __getitem__(self, item):
        fname = self.ds[item]

        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, -1)
        image,float_image = self.preprocess_func(image)

        return fname, image,float_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights=['./models/fold0_epoch_14_val_loss0.368938.pth']

test_datadir='../cassava-leaf-disease-classification/test_images'

dataiter=DatasetTest(test_datadir)

def predict_with_model(model,weights):

    for weight in weights:

        model.load_state_dict(torch.load(weight, map_location=device))
        ### load your weights
        model.eval()

        cur_result=pd.DataFrame(columns=['image_id','label'])

        image_ids=[]
        precictions=[]
        for i in range(len(dataiter)):
            fname,_,float_image=dataiter.__getitem__(i)

            input=torch.from_numpy(float_image).to(device).float()
            with torch.no_grad():
                output=model(input)

                output=torch.nn.functional.softmax(output,dim=-1)
                output=output.cpu().numpy()

                output=np.mean(output,axis=0)

            image_ids.append(fname)

            label=np.argmax(output)
            precictions.append(label)

        cur_result['image_id']=image_ids
        cur_result['label'] = precictions

        cur_result.to_csv(weight+'.csv',index=False)
        cur_result.to_csv('submission.csv',index=False)


model=Net().to(device)


predict_with_model(model,weights)