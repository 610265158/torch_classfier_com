# Dice损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Model):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, targets,weights):

        input=torch.sigmoid(input)
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)*weights
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss