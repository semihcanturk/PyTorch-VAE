import yaml
import argparse
import numpy as np
import matplotlib as plt

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
                    default='./configs/cvae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])

model_dict = torch.load('celeba-epoch=07-val_loss=11.28.ckpt')
newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v

model.load_state_dict(newdict)

model.eval()
input_vect = torch.zeros(config['exp_params']['batch_size'], config['model_params']['num_classes'])
out = model.encode(input_vect)



# feature_idx = np.random.randint(0, config['model_params']['num_classes'])
# input_vect = torch.zeros(config['exp_params']['batch_size'], config['model_params']['num_classes'])
# input_vect[:, feature_idx] = 1
# out = model.infer(input_vect)
# vutils.save_image(out.cpu().data, f"outputs/"f"feature_{feature_idx}.png", normalize=True, nrow=12)


