import sys

sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from timm.models.layers.activations import HardMish
from train_config import config as cfg

import timm

BN_momentum=0.03
BN_eps=1e-5
class ComplexUpsample(nn.Module):
    def __init__(self, input_dim=128, outpt_dim=128):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, outpt_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outpt_dim,momentum=BN_momentum,eps=BN_eps),
                                   HardMish()
                                   )

    def forward(self, x):
        # do preprocess

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv1(x)
        return x


class Fpn(nn.Module):
    def __init__(self, input_dims=[40, 64, 176, 512], head_dims=[256, 256, 256]):
        super().__init__()

        self.latlayer2 = nn.Sequential(nn.Conv2d(input_dims[0], head_dims[0] , kernel_size=3, padding=1),
                                       nn.BatchNorm2d(head_dims[0] ,momentum=BN_momentum,eps=BN_eps),
                                       HardMish())

        self.latlayer3 = nn.Sequential(nn.Conv2d(input_dims[1], head_dims[1], kernel_size=3, padding=1),
                                       nn.BatchNorm2d(head_dims[1] ,momentum=BN_momentum,eps=BN_eps),
                                       HardMish())

        self.latlayer4 = nn.Sequential(nn.Conv2d(input_dims[2], head_dims[2] , kernel_size=3, padding=1),
                                       nn.BatchNorm2d(head_dims[2] ,momentum=BN_momentum,eps=BN_eps),
                                       HardMish())

        self.upsample3 = ComplexUpsample(head_dims[1], head_dims[0] )

        self.upsample4 = ComplexUpsample(head_dims[2], head_dims[1] )

        self.upsample5 = ComplexUpsample(input_dims[3], head_dims[2] )

    def forward(self, inputs):
        ##/24,32,96,320
        c2, c3, c4, c5 = inputs

        c4_lat = self.latlayer4(c4)
        c3_lat = self.latlayer3(c3)
        c2_lat = self.latlayer2(c2)

        upsample_c5 = self.upsample5(c5)

        p4 = c4_lat+ upsample_c5

        upsample_p4 = self.upsample4(p4)

        p3 = c3_lat+upsample_p4

        upsample_p3 = self.upsample3(p3)

        p2 = c2_lat+upsample_p3

        return p2


class Net(nn.Module):
    def __init__(self, num_classes=cfg.MODEL.num_class):
        super().__init__()

        # self.mean_tensor=torch.from_numpy(cfg.DATA.PIXEL_MEAN ).float().cuda()
        # self.std_val_tensor = torch.from_numpy(cfg.DATA.PIXEL_STD).float().cuda()
        # self.model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        self.model = timm.create_model('seresnet152d', pretrained=False, features_only=True)
        # self.model = timm.create_model('hrnet_w32', pretrained=True)

        self.fpn = Fpn()

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)

        self._fc = nn.Linear(512+11, num_classes, bias=True)

        self.seg = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, padding=0, stride=1,bias=True)

    def forward(self, inputs):
        # do preprocess


        bs = inputs.size(0)
        # Convolution layers
        # x = self.model.forward_features(inputs)
        fms = self.model(inputs)
        for x in fms:
            print(x.size())

        x = fms[-1]

        fm = self._avg_pooling(x)
        fm = fm.view(bs, -1)
        feature = self.dropout(fm)

        seg = self.fpn(fms[1:])
        seg = self.seg(seg)

        segfm = self._avg_pooling(seg)
        segfm = segfm.view(bs, -1)

        feature=torch.cat([feature,segfm],dim=1)

        x = self._fc(feature)
        return x, seg


if __name__ == '__main__':
    import torch
    import torchvision

    dummy_input = torch.randn(1, 3, 512, 512, device='cpu')
    model = Net()

    ### load your weights
    model.eval()
    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.

    torch.onnx.export(model, dummy_input, "classifier.onnx")

