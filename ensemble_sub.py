
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd


from make_data import Tokenizer
from train_config import config as cfg
import albumentations as A

import setproctitle


setproctitle.setproctitle("inference")

from lib.core.base_trainer.model import Caption
import torch
from tqdm import tqdm
import gc
weights=['./models/fold0_epoch_0_val_dis_2.165299_loss_0.588935.pth',
         './models/fold0_epoch_6_val_dis_2.661832_loss_0.749578.pth']

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 200

attention_dim = 300
encoder_dim = 512
decoder_dim = 300

def get_test_df():
    test = pd.read_csv('oritation_submission.csv')

    def get_test_file_path(image_id):
        return "../test/test/{}/{}/{}/{}.png".format(
            image_id[0], image_id[1], image_id[2], image_id
        )

    test['file_path'] = test['image_id'].apply(get_test_file_path)
    return test


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

    def letterbox(self, image, target_shape=(cfg.MODEL.height, cfg.MODEL.width)):

        h, w = image.shape

        if h / w >= target_shape[0] / target_shape[1]:
            size = (h, int(1.5 * h))

            image_container = np.zeros(shape=size, dtype=np.uint8) + 255

            image_container[:, (size[1] - w) // 2:(size[1] - w) // 2 + w] = image
        else:
            size = (int(w / 1.5), w)

            image_container = np.zeros(shape=size, dtype=np.uint8) + 255

            image_container[(size[0] - h) // 2:(size[0] - h) // 2 + h, :] = image

        return image_container
    def single_map_func(self, dp):
        """Data augmentation function."""
        ####customed here
        fname = dp['file_path']
        oritation =int(dp['oritation'])
        image_raw=cv2.imread(fname,-1)

        edge = 25
        image_raw = image_raw[edge:-edge, edge:-edge]

        h,w=image_raw.shape
        if oritation==1 or h / w>1.5:
            image_raw = self.fix_transform(image=image_raw)['image']

        image_raw=self.letterbox(image_raw)

        transformed = self.val_trans(image=image_raw)

        image = transformed['image']
        image = np.expand_dims(image,0)
        return 255-image

def get_dataiter(df):


    test_generator = TestDataIter(df)

    test_ds = DataLoader(test_generator,
                             cfg.TRAIN.batch_size * 3+64,
                             num_workers=cfg.TRAIN.process_num+2,
                             shuffle=False)

    return test_ds



def main():

    result_f=open('tmp.txt','w')
    test = get_test_df()

    text_preds = []
    ##内存问题多fold，分开跑
    n_fold = 2
    n_slice=50

    _slice=len(test)//n_slice

    for i in range(n_slice):
        if i!=n_slice-1:
            cur_df=test[i*_slice:(i+1)*_slice]
        else:
            cur_df=test[i*_slice:]
        test_ds = get_dataiter(cur_df)


        all_fold_predictions=[]
        for fold in range(n_fold):
            print(' infer with fold %d,  slice %d' % (fold, i))
            model.load_state_dict(torch.load(weights[fold], map_location=device))

            model.eval()

            one_fold_prediction=[]

            with torch.no_grad():
                for images in tqdm(test_ds):
                    data = images.to(device).float()

                    predictions = model(data)

                    predictions = torch.softmax(predictions,dim=-1).cpu().numpy().astype(np.float16)
                    one_fold_prediction.append(predictions)
            one_fold_prediction=np.concatenate(one_fold_prediction)
            all_fold_predictions.append(one_fold_prediction)

            del one_fold_prediction
        ### predict
        predictions=np.mean(all_fold_predictions,axis=0)
        predicted_sequence = np.argmax(predictions, -1)


        batch_size=predicted_sequence.shape[0]

        for k in range(batch_size):
            caption = token_tools.predict_caption(predicted_sequence[k])
            result_f.write(caption + '\n')

        del predicted_sequence
        del predictions
        del all_fold_predictions
        gc.collect()


    result_f.close()
    with open('./tmp.txt','r') as f:
        text_preds=f.readlines()

    test['InChI'] = [f"InChI=1S/{text.rstrip()}" for text in text_preds]
    test[['image_id', 'InChI']].to_csv('submission.csv', index=False)
    test[['image_id', 'InChI']].head()



if __name__=='__main__':
    main()