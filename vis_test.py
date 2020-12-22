import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
from torchvision.datasets import CIFAR10, CelebA



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

params = config['exp_params']

model = vae_models[config['model_params']['name']](**config['model_params'])

model_dict = torch.load('/Users/semo/Desktop/CKPT BABY/CelebA-VAE-LD128-LR00001-epoch=49-val_loss=13.74.ckpt-epoch=49-val_loss=34.77.ckpt',map_location=torch.device('cpu'))

newdict = dict()
for k, v in model_dict['state_dict'].items():
    knew = k.split('.', 1)[1:][0]
    newdict[knew] = v


model.load_state_dict(newdict)
model.eval()


def data_transforms():
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

    if params['dataset'] == 'cifar10':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(params['img_size']),
                                        transforms.ToTensor(),
                                        SetRange])
    elif params['dataset'] == 'celeba':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(params['img_size']),
                                        transforms.ToTensor(),
                                        SetRange])
    elif params['dataset'] == 'wikiart':
        transform = transforms.Compose([transforms.Resize(64),
                                        transforms.RandomCrop(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    else:
        raise ValueError('Undefined dataset type')
    return transform



def get_dataloader():
    transform = data_transforms()

    if params['dataset'] == 'cifar10':
        sample_dataloader = DataLoader(CIFAR10(root=params['data_path'],
                                                        train=False,
                                                        transform=transform,
                                                        download=False),
                                                batch_size=144,
                                                num_workers=12,
                                                shuffle=True,
                                                drop_last=True)
        num_val_imgs = len(sample_dataloader)
    elif params['dataset'] == 'celeba':
        sample_dataloader = DataLoader(CelebA(root=params['data_path'],
                                                       split="test",
                                                       transform=transform,
                                                       download=False),
                                                num_workers=12,
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)
        num_val_imgs = len(sample_dataloader)
    elif params['dataset'] == 'wikiart':
            main_dataset = datasets.ImageFolder(root=params['data_path'] + 'wikiart',
                                                transform=transform)
            train_size = int(0.8 * len(main_dataset))
            test_size = len(main_dataset) - train_size
            _, test_dataset = torch.utils.data.random_split(main_dataset, [train_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
            sample_dataloader = DataLoader(test_dataset,
                                                num_workers=12,
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)
            num_val_imgs = len(sample_dataloader)
    else:
        raise ValueError('Undefined dataset type')

    return sample_dataloader

try:
    embed = np.load('vis_embed_arr.npy')
    label_emb = np.load('vis_label_embed_arr.npy')
except:
    dataloader = get_dataloader()
    embed = []
    label_emb = []
    for inputs, labels in dataloader:
        with torch.no_grad():
            if config['exp_params']['dataset'] == 'wikiart':
                num_classes = 27
            else:
                num_classes = -1
            # for the colors
            label_emb.append(labels.numpy())
            # for the embedding
            labels = torch.nn.functional.one_hot(labels, num_classes)
            embed.append(model.embed(inputs, labels = labels).numpy())

    embed = np.array(embed).reshape(-1,model.latent_dim)
    label_emb = np.array(label_emb)
    np.save('vis_embed_arr', embed)
    np.save('vis_label_embed_arr', label_emb)

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
    try:
        embed_tsne = np.load('VAE_TSNE_128.npy')
    except:
        embed_tsne = TSNE(n_components=2).fit_transform(embed)
        np.save('VAE_TSNE_128', embed_tsne)

    class_labels = np.argmax(label_emb.reshape(-1,label_emb.shape[-1]), axis=1)
    plt.scatter(embed_tsne[:,0],embed_tsne[:,1], c=class_labels, s=1, cmap='flag')
    plt.savefig('embed'+config['model_params']['name']+str(config['model_params']['latent_dim'])+'LD'+'.png')
    #plt.savefig('embed'+config['logging_params']['name']+'.png')

