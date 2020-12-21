import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
import torchvision.utils as vutils

import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms, datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from PIL import ImageFile
from sklearn.manifold import TSNE



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

model_dict = torch.load('celeba-epoch=21-val_loss=11.88.ckpt',map_location=torch.device('cpu'))

newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v


model.load_state_dict(newdict)
model.eval()

transform = transforms.Compose([transforms.Resize(64),
                                transforms.RandomCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
dataset = datasets.ImageFolder(root= "/Users/GuillaumeHuguetLarrieu/PycharmProjects/PyTorch-VAE/data/" + 'wikiart', transform=transform)

#we could use a subsample

dataloader = DataLoader(dataset,
                                                num_workers=12,
                                                batch_size=1,
                                                shuffle=True,
                                                drop_last=True)

embed = []
label_emb = []
for inputs, labels in dataloader:
    with torch.no_grad():
        if config['exp_params']['dataset'] == 'wikiart':
            num_classes = 27
        else:
            num_classes = -1
        # for the colors
        label_emb.append(labels)
        # for the embedding
        labels = torch.nn.functional.one_hot(labels, num_classes)
        embed.append(model.embed(inputs, labels = labels).numpy())

embed = np.array(embed).reshape(-1,model.latent_dim)

if model.latent_dim == 2:
    plt.scatter(embed[:,0],embed[:,1], c  = label_emb)
    plt.savefig('embed'+config['model_params']['name']+str(config['model_params']['latent_dim'])+'LD'+'.png')
    #plt.savefig('embed'+config['logging_params']['name']+'.png')
elif model.latent_dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2], c = label_emb)
    ax.view_init(10, 80)
    plt.savefig('embed'+config['model_params']['name']+str(config['model_params']['latent_dim'])+'LD'+'.png')
    #plt.savefig('embed'+config['logging_params']['name']+'.png')
else:
    embed_tsne = TSNE(n_components=2).fit_transform(embed)
    plt.scatter(embed_tsne[:,0],embed_tsne[:,1], c  =  np.array(label_emb).reshape(-1,1))
    plt.savefig('embed'+config['model_params']['name']+str(config['model_params']['latent_dim'])+'LD'+'.png')
    #plt.savefig('embed'+config['logging_params']['name']+'.png')

