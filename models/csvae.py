import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
import torch.distributions as dists
import numpy as np
from .types_ import *


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class Conv_block(nn.Module):
    def __init__(self, num_features, in_channels, out_channels, kernel_size, stride=1, padding=0, negative_slope=0.2,
                 p=0.04, transpose=False):
        super(Conv_block, self).__init__()

        self.transpose = transpose
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        self.dropout = nn.Dropout2d(p)
        self.batch_norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if not self.transpose:
            x = self.dropout(x)
        x = self.batch_norm(x)

        return x


class CSVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
                 **kwargs) -> None:
        super(CSVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.w_dim = 2
        p = 0.04

        # x to x_features_dim
        self.encoder = nn.Sequential()
        self.encoder.add_module("block01", Conv_block(self.img_size, 3, self.img_size, 4, 2, 1, p=p))
        self.encoder.add_module("block02", Conv_block(self.img_size * 2, self.img_size, self.img_size * 2, 4, 2, 1, p=p))
        self.encoder.add_module("block03", Conv_block(self.img_size * 4, self.img_size * 2, self.img_size * 4, 4, 2, 1, p=p))
        self.encoder.add_module("block04", Conv_block(self.img_size * 8, self.img_size * 4, self.img_size * 8, 4, 2, 1, p=p))
        self.encoder.add_module("block05", Conv_block(self.img_size * 16, self.img_size * 8, self.img_size * 16, 4, 2, 1, p=p))
        self.encoder.add_module("block06", Conv_block(self.img_size * 16, self.img_size * 16, self.img_size * 16, 4, 2, 1, p=p))
        #         self.encoder.add_module("block07", Conv_block(self.img_size*32, self.img_size*16, self.img_size*32, 4, 2, 1, p=p))
        #         self.encoder.add_module("block08", Conv_block(self.img_size*32, self.img_size*32, self.img_size*32, 4, 2, 1, p=p))
        self.encoder.add_module("flatten", Flatten())

        x_features_dim = self.img_size * 8 * 2

        self.encoder_xy_to_w = nn.Sequential(
            nn.Linear(x_features_dim + num_classes, self.w_dim),
            nn.ReLU(),
        )
        self.mu_xy_to_w = nn.Linear(self.w_dim, self.w_dim)
        self.logvar_xy_to_w = nn.Linear(self.w_dim, self.w_dim)

        self.encoder_x_to_z = nn.Sequential(
            nn.Linear(x_features_dim, self.latent_dim),
            nn.ReLU(),
        )
        self.mu_x_to_z = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_x_to_z = nn.Linear(self.latent_dim, self.latent_dim)

        self.encoder_y_to_w = nn.Sequential(
            nn.Linear(num_classes, self.w_dim),
            nn.ReLU(),
            #             nn.Linear(self.w_dim, self.w_dim),
            #             nn.ReLU()
        )
        self.mu_y_to_w = nn.Linear(self.w_dim, self.w_dim)
        self.logvar_y_to_w = nn.Linear(self.w_dim, self.w_dim)

        # Add sigmoid or smth for images!
        # (z+w) to x_sample
        # (!) no logvar for x
        self.decoder_zw_to_x = nn.Sequential()
        self.decoder_zw_to_x.add_module("block00", nn.Sequential(
            nn.Linear(self.latent_dim + self.w_dim, self.latent_dim + self.w_dim),
            nn.BatchNorm1d(self.latent_dim + self.w_dim),
            nn.LeakyReLU(0.2)
        ))
        self.decoder_zw_to_x.add_module("reshape", Reshape((-1, self.latent_dim + self.w_dim, 1, 1)))

        self.decoder_zw_to_x.add_module("block01",
                                        Conv_block(self.img_size * 4, self.latent_dim + self.w_dim, self.img_size * 4, 4, 1, 0, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block02", Conv_block(self.img_size * 4, self.img_size * 4, self.img_size * 4, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block03", Conv_block(self.img_size * 2, self.img_size * 4, self.img_size * 2, 3, 1, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block04", Conv_block(self.img_size * 2, self.img_size * 2, self.img_size * 2, 4, 2, 1, p=p, transpose=True))
        #         self.decoder_zw_to_x.add_module("block05", Conv_block(self.img_size*4, self.img_size*4, self.img_size*4, 4, 2, 1, p=p, transpose=True))
        #         self.decoder_zw_to_x.add_module("block06", Conv_block(self.img_size*2, self.img_size*4, self.img_size*2, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block05", Conv_block(self.img_size, self.img_size * 2, self.img_size, 4, 2, 1, p=p, transpose=True))
        self.decoder_zw_to_x.add_module("block06", Conv_block(self.img_size, self.img_size, self.img_size, 4, 2, 1, p=p, transpose=True))
        #         self.decoder_zw_to_x.add_module("block07", nn.Sequential(
        #                     nn.ConvTranspose2d(self.img_size, 3, 3, 1, 1)))

        self.mu_zw_to_x = nn.Sequential(
            nn.ConvTranspose2d(self.img_size, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.logvar_zw_to_x = nn.Sequential(
            nn.ConvTranspose2d(self.img_size, 3, 3, 1, 1),
            #             nn.Tanh()
        )
        #         self.logvar_zw_to_x = nn.Linear(self.latent_dim+self.w_dim, input_dim)

        # adversarial delta(z -> y)
        self.decoder_z_to_y = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, num_classes),
            nn.Sigmoid()
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def q_zw(self, x, y):
        """
        VARIATIONAL POSTERIOR
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """

        x_features = self.encoder(x)

        intermediate = self.encoder_x_to_z(x_features)
        z_mu = self.mu_x_to_z(intermediate)
        z_logvar = self.logvar_x_to_z(intermediate)

        xy = torch.cat([x_features, y], dim=1)


        intermediate = self.encoder_xy_to_w(xy)
        w_mu_encoder = self.mu_xy_to_w(intermediate)
        w_logvar_encoder = self.logvar_xy_to_w(intermediate)

        intermediate = self.encoder_y_to_w(y)
        w_mu_prior = self.mu_y_to_w(intermediate)
        w_logvar_prior = self.logvar_y_to_w(intermediate)

        return w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar

    def p_x(self, z, w):
        """
        GENERATIVE DISTRIBUTION
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """

        zw = torch.cat([z, w], dim=1)

        intermediate = self.decoder_zw_to_x(zw)
        mu = self.mu_zw_to_x(intermediate)
        logvar = self.logvar_zw_to_x(intermediate)

        return mu, logvar


    def encode(self, input: Tensor, y: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        w_mu_encoder, w_logvar_encoder, w_mu_prior, \
        w_logvar_prior, z_mu, z_logvar = self.q_zw(input, y)
        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        w_prior = self.reparameterize(w_mu_prior, w_logvar_prior)

        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.p_x(z, w_encoder)

        return [input, z, w_mu_encoder, w_logvar_encoder, w_mu_prior,
                w_logvar_prior, z_mu, z_logvar, w_encoder, w_prior, x_mu, x_logvar]

    def decode(self, input, z, w_mu_encoder, w_logvar_encoder, w_mu_prior,
                w_logvar_prior, z_mu, z_logvar, w_encoder, w_prior, x_mu, x_logvar) -> Tensor:
        # input = z
        zw = torch.cat([z, w_encoder], dim=1)  # for adversarial train

        y_pred = self.decoder_z_to_y(z)  # for adversarial train

        return input, x_mu, x_logvar, zw, y_pred, \
               w_mu_encoder, w_logvar_encoder, w_mu_prior, \
               w_logvar_prior, z_mu, z_logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        y = kwargs['labels'].float()
        #embedded_class = self.embed_class(y)
        #embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        #embedded_input = self.embed_data(input)

        #x = torch.cat([embedded_input, embedded_class], dim = 1)
        input, z, w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar, w_encoder, w_prior, x_mu, \
        x_logvar = self.encode(input, y)

        return self.decode(input, z, w_mu_encoder, w_logvar_encoder, w_mu_prior,
                w_logvar_prior, z_mu, z_logvar, w_encoder, w_prior, x_mu, x_logvar)

    def infer(self, y, z=None):
        # TODO
        if z is None:
            z = torch.randn(y.shape[0], self.latent_dim)
            z = torch.cat([z, y], dim=1)
        return self.decode(z)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        x = args[0]
        x_mu = args[1]
        x_logvar = args[2]
        zw = args[3]
        y_pred = args[4]
        w_mu_encoder = args[5]
        w_logvar_encoder = args[6]
        w_mu_prior = args[7]
        w_logvar_prior = args[8]
        z_mu = args[9]
        z_logvar = args[10]

        beta1 = 20
        beta2 = 1
        beta3 = 0.2
        beta4 = 10
        beta5 = 1

        recons_loss = nn.MSELoss()(x_mu, x)

        # w_kl

        w_mu_np = np.array(w_mu_encoder.cpu().detach())
        w_logvar_np = np.array(w_logvar_encoder.cpu().detach())

        if np.isnan(w_mu_np).any():
            np.save('w_mu', w_mu_np)
        if np.isnan(w_logvar_np).any():
            np.save('w_logvar', w_logvar_np)

        w_dist = dists.MultivariateNormal(w_mu_encoder.flatten(),
                                          torch.diag(w_logvar_encoder.flatten().exp()))
        w_prior = dists.MultivariateNormal(w_mu_prior.flatten(),
                                           torch.diag(w_logvar_prior.flatten().exp()))
        w_kl = dists.kl.kl_divergence(w_dist, w_prior)

        # z_kl
        z_dist = dists.MultivariateNormal(z_mu.flatten(),
                                          torch.diag(z_logvar.flatten().exp()))
        z_prior = dists.MultivariateNormal(torch.zeros(self.latent_dim * z_mu.size()[0]).to(z_mu),
                                           torch.eye(self.latent_dim * z_mu.size()[0]).to(z_mu))
        z_kl = dists.kl.kl_divergence(z_dist, z_prior)

        # -H(y)
        y_pred_negentropy = (y_pred.log() * y_pred + (1 - y_pred).log() * (1 - y_pred)).mean()

        # y xentropy
        # y_recon = nn.BCELoss()(y_pred, y)  # alternatively use predicted logvar too to evaluate density of input

        ELBO = beta1 * recons_loss + beta2 * w_kl + beta3 * z_kl + beta4 * y_pred_negentropy
        return {'loss': ELBO, 'Reconstruction_Loss':recons_loss, 'KLD':-(z_kl + w_kl)}


    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[1]

