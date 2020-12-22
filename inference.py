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
                    default='./configs/vae_vis.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = vae_models[config['model_params']['name']](**config['model_params'])

model_dict = torch.load('/Users/semo/Desktop/CKPT BABY/VanillaVAE (CelebA, LD=40, LR=0.001).ckpt', map_location=torch.device('cpu'))
newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v

model.load_state_dict(newdict)


model.eval()

#feature_idx = np.random.randint(0, config['model_params']['num_classes'])
def plot_cvae_classes():
    for feature_idx in range(config['model_params']['num_classes']):
        input_vect = torch.zeros(config['exp_params']['batch_size'], config['model_params']['num_classes'])
        input_vect[:, feature_idx] = 1
        out = model.infer(input_vect)
        vutils.save_image(out.cpu().data, f"outputs/"f"feature_{feature_idx}.png", normalize=True, nrow=12)


def plot_latent_space(dim1=None, dim2=None, sample_dim=20, dir='outputs/latent_space/'):
    if dim1 is None:
        dim1 = np.random.randint(0, config['model_params']['latent_dim'])
    if dim2 is None:
        dim2 = np.random.randint(0, config['model_params']['latent_dim'])

    grid = model.plot_latent_space(dim1, dim2, sample_dim)
    dir = dir + f"{config['exp_params']['dataset']}_{config['model_params']['name']}_LD={config['model_params']['latent_dim']}/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    vutils.save_image(grid, dir + f"latent_space_{dim1},{dim2}.png", normalize=True, nrow=sample_dim)

def plot_all_latent_spaces():
    for i in range(config['model_params']['latent_dim']):
        for j in range(config['model_params']['latent_dim']):
            if i < j:
                plot_latent_space(i, j)

def plot_random_latent_spaces(num_samples=20):
    for i in range(num_samples):
        plot_latent_space()

plot_random_latent_spaces(100)



