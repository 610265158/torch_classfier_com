import Levenshtein

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

class DISTANCEMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):

        self.y_true = None
        self.y_pred = None
        self.scores=[]

    def update(self, y_true, y_pred):

        for i in range(y_true.shape[0]):

            score=Levenshtein.distance(y_true[i],y_pred[i])
            
            self.scores.append(score)




    @property
    def avg(self):

        return np.mean(self.scores)



