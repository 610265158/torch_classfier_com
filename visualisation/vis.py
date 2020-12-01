import sys
sys.path.append('.')
import os
import numpy as np
import cv2

import torch.utils.data


from lib.core.base_trainer.model import Net



data_dir = '../picture_scripts/sample/美食'

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
        image=cv2.resize(image,(224,224))
        image_batch = np.stack([image])

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        # image = np.expand_dims(image, 0)
        return image, image_batch

    def __getitem__(self, item):
        fname = self.ds[item]

        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, -1)
        image,float_image = self.preprocess_func(image)

        return fname, image,float_image



weights='fold4_epoch_10_val_loss0.029808.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Net()
model.load_state_dict(torch.load(weights, map_location=device))

### load your weights
model.eval()




dataiter=DatasetTest(data_dir)


for i in range(len(dataiter)):
    fname,image,float_image=dataiter.__getitem__(i)


    input=torch.from_numpy(float_image)

    res=model(input)

    res=torch.sigmoid(res)

    print(res)
    cv2.imshow('ss',image)
    cv2.waitKey(0)