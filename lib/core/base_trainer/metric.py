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


import warnings

warnings.filterwarnings('ignore')

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






class ROCAUCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):

        self.y_true_11=None
        self.y_pred_11 = None

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()

        y_pred = torch.sigmoid(y_pred).data.cpu().numpy()



        if self.y_true_11 is None:
            self.y_true_11 = y_true
            self.y_pred_11 = y_pred
        else:
            self.y_true_11 = np.concatenate((self.y_true_11, y_true),axis=0)
            self.y_pred_11 = np.concatenate((self.y_pred_11, y_pred),axis=0)

    def fast_auc(self,y_true, y_prob):
        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_prob)]
        cumfalses = np.cumsum(1 - y_true)
        nfalse = cumfalses[-1]
        auc = (y_true * cumfalses).sum()
        auc /= (nfalse * (len(y_true) - nfalse))
        return auc

    @property
    def avg(self):

        score=self.fast_auc(self.y_true_11,self.y_pred_11)

        
        return score



