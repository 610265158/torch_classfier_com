import sys
sys.path.append('.')

from lib.core.base_trainer.model import Net


import torch
import torchvision
import argparse


from  train_config import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,default=None, help='the thres for detect')
args = parser.parse_args()

model_path=args.model



weights='./fold4_epoch_6_val_loss0.031719.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
dummy_input = torch.randn(1, cfg.MODEL.channel, cfg.MODEL.height, cfg.MODEL.width, device='cpu')

model = Net()
if model_path is not None:
    model.load_state_dict(torch.load(weights, map_location=device))

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


torch.onnx.export(model, dummy_input, "scene_classifier.onnx" ,
                  input_names=["input"], output_names=["output"])
