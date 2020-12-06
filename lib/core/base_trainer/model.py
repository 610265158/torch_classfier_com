
import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from efficientnet_pytorch import EfficientNet
from train_config import config as cfg


import timm
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
    def __init__(self, num_classes=cfg.MODEL.num_class):
        super().__init__()

        # self.mean_tensor=torch.from_numpy(cfg.DATA.PIXEL_MEAN ).float().cuda()
        # self.std_val_tensor = torch.from_numpy(cfg.DATA.PIXEL_STD).float().cuda()
        # self.model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        # self.model = timm.create_model('mobilenetv2_110d', pretrained=True)

        # self.model = timm.create_model('mobilenetv2_110d', pretrained=True)
        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.dropout=nn.Dropout(0.5)

        self._fc = nn.Linear(1280 , num_classes, bias=True)

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



if __name__=='__main__':
    import torch
    import torchvision

    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
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


    torch.onnx.export(model, dummy_input, "classifier.onnx" )

