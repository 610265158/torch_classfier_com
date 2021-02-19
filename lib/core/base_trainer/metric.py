import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import numpy as np
import torch.nn as nn
from train_config import config as cfg

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from lib.utils.logger import logger
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ACCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0,1])
        self.y_pred = np.array([0,1])
        self.score = 0



        self.y_true_11=np.array([[0,0,0,0,0,0,0,0,0,0,0]])
        self.y_pred_11 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()

        y_pred = torch.nn.functional.softmax(y_pred).data.cpu().numpy()

        self.y_true_11 = np.concatenate((self.y_true_11, y_true),axis=0)
        self.y_pred_11 = np.concatenate((self.y_pred_11, y_pred),axis=0)



        y_true = np.argmax(y_true, 1)

        y_pred = np.argmax(y_pred,1)

        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))


    @property
    def avg(self):
        right=(self.y_pred==self.y_true).astype(np.float)

        auc_scor=0

        for i in range(11):
            try:
                tmp_score=roc_auc_score(self.y_true_11[:, i], self.y_pred_11[:, i])
            except:
                tmp_score=0

            auc_scor+=tmp_score/11.
        return np.sum(right)/self.y_true.shape[0],auc_scor

