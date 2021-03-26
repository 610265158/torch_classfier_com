
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd


from make_data import Tokenizer
from train_config import config as cfg
import albumentations as A

import setproctitle


setproctitle.setproctitle("comp")

from lib.core.base_trainer.model import Caption
import torch
from tqdm import tqdm

weights=['./models/fold0_epoch_0_val_acc_0.999256_loss_0.001846.pth']

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 200

attention_dim = 300
encoder_dim = 512
decoder_dim = 300

def get_test_df():
    test = pd.read_csv('../test/sample_submission.csv')

    def get_test_file_path(image_id):
        return "../test/test/{}/{}/{}/{}.png".format(
            image_id[0], image_id[1], image_id[2], image_id
        )

    test['file_path'] = test['image_id'].apply(get_test_file_path)
    return test

test=get_test_df()

def get_token():
    token_tools = Tokenizer()
    token_tools.stoi = np.load("../tokenizer.stio.npy", allow_pickle=True).item()
    token_tools.itos = np.load("../tokenizer.itos.npy", allow_pickle=True).item()

    return token_tools


token_tools = get_token()

def get_model():
    model = Caption(embed_dim=embed_dim,
                    vocab_size=len(token_tools),
                    attention_dim=attention_dim,
                    encoder_dim=encoder_dim,
                    decoder_dim=decoder_dim,
                    dropout=0.5,
                    max_length=cfg.MODEL.train_length - 1,
                    tokenizer=token_tools).to(device)

    return model

model=get_model()


class TestDataIter():
    def __init__(self, df):

        self.df = df

        self.val_trans=A.Compose([

                                   A.Resize(height=cfg.MODEL.height,
                                           width=cfg.MODEL.width)

                                  ])

        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item])

    def __len__(self):


        return len(self.df)

    def letterbox(self,image):
        h,w=image.shape

        size=max(h,w)
        image_container=np.zeros(shape=[size,size],dtype=np.uint8)+255

        image_container[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w]=image

        return image_container
    def single_map_func(self, dp):
        """Data augmentation function."""
        ####customed here
        fname = dp['file_path']

        image_raw=cv2.imread(fname,-1)
        h,w=image_raw.shape
        if h > w:
            image_raw = self.fix_transform(image=image_raw)['image']


        image_raw=self.letterbox(image_raw)

        transformed = self.val_trans(image=image_raw)

        image = transformed['image']
        image = np.stack([image, image, image], 0)
        return 255-image

def get_dataiter():


    test_generator = TestDataIter(test)

    test_ds = DataLoader(test_generator,
                             cfg.TRAIN.batch_size * 2,
                             num_workers=cfg.TRAIN.process_num,
                             shuffle=False)

    return test_ds

test_ds=get_dataiter()

def main():
    n_fold=5
    text_preds = []

    pres = []
    for fold in range(1):
        print(' infer with fold %d' % (fold))
        model.load_state_dict(torch.load(weights[fold], map_location=device))

        model.eval()

        one_fold_predictin=[]
        with torch.no_grad():
            for images in tqdm(test_ds):
                data = images.to(device).float()

                predictions = model(data)
                predictions = torch.softmax(predictions,dim=-1).cpu().numpy()

                one_fold_predictin.append(predictions)

        one_fold_predictin=np.concatenate(one_fold_predictin)
        pres.append(one_fold_predictin)

    predictions=np.mean(pres,axis=0)
    ### predict
    oritation = np.argmax(predictions, -1)



    test['oritation'] = oritation
    test[['image_id', 'InChI','oritation']].to_csv('oritation_submission.csv', index=False)
    test.head()



if __name__=='__main__':
    main()