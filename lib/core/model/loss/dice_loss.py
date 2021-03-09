import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()


        self.dice_func= DiceLoss()
    def forward(self, input, target, weights=None):

        if weights is not None:
            input=input[weights,...]
            target = target[weights, ...]

        C = target.shape[1]



        totalLoss = 0

        for i in range(C):
            diceLoss =  self.dice_func(input[:, i], target[:, i])

            totalLoss += diceLoss

        return totalLoss
