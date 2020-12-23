import yaml
import argparse
import numpy as np
import os

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
import torchvision.utils as vutils


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='./configs/cvae_vis.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])

model_dict = torch.load('/Users/semo/Desktop/CKPT BABY/CelebA-CVAE-LD=40-LR=0.0001-epoch=49-val_loss=31.57.ckpt', map_location=torch.device('cpu'))
newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v

model.load_state_dict(newdict)


model.eval()

import csv

with open('features_list.txt', mode='r') as infile:
    reader = csv.reader(infile)
    feature_dict = dict()
    i=0
    for row in reader:
        if len(row) > 0:
            feature_dict[i] = row[0].strip()
            i += 1

def plot_cvae_vector(idxs):
    input_vect = torch.zeros(config['exp_params']['batch_size'], config['model_params']['num_classes'])
    filename = ""
    for feature_idx in idxs:
        input_vect[:, feature_idx] = 1
        filename += feature_dict[feature_idx] + "_"
    out = model.infer(input_vect)
    vutils.save_image(out.cpu().data, f"outputs/"f"{filename}.png", normalize=True,
                      nrow=12)

#feature_idx = np.random.randint(0, config['model_params']['num_classes'])
def plot_cvae_classes():
    for feature_idx in range(config['model_params']['num_classes']):
        input_vect = torch.zeros(config['exp_params']['batch_size'], config['model_params']['num_classes'])
        input_vect[:, feature_idx] = 1
        out = model.infer(input_vect)
        vutils.save_image(out.cpu().data, f"outputs/"f"{feature_idx}_{feature_dict[feature_idx]}.png", normalize=True, nrow=12)

plot_cvae_classes()
plot_cvae_vector([0, 4, 7, 13, 14, 15, 16, 31])
plot_cvae_vector([5, 21, 23, 25])
plot_cvae_vector([2, 17, 20])
plot_cvae_vector([2, 6, 8, 27, 29, 31, 39])




