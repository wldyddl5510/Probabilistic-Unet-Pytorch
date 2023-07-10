#This code is based on: https://github.com/SimonKohl/probabilistic_unet

from unet_blocks import *
from unet import Unet
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from distributions import HypersphericalUniform, VonMisesFisher

# escnn library for implementing equivariant CNN -> This will be used to make Kendall shape space embedding
from escnn import gspaces
from escnn import nn as escnn_nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

"""
escnn encoder to encode equivariant layer
Use N = 8 for temporary
"""
class EquiEncoder(Encoder):


    def __init__(self, input_channels, num_filters, no_convs_per_block, initializer, padding = True, posterior = False):
        super().__init__(input_channels, num_filters, no_convs_per_block, initializer, padding, posterior)
        # define equivariant encoder
        # code from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html#3.-Build-and-Train-Steerable-CNNs
        # rotation axis N = 8 stands for 8 directions
        self.r2_act = gspaces.rot2dOnR2(N = 8)
        in_type = escnn_nn.FieldType(self.r2_act, self.input_channels * [self.r2_act.trivial_repr])
        self.input_type = in_type

        # construct layers
        layers = []
        for i in range(len(self.num_filters)):
            #TODO: implement equivariant encoder
            out_type = escnn_nn.FieldType(self.r2_act, num_filters[i] * [self.r2_act.regular_repr])
            # other than 1st layer
            if i != 0:
                in_type = layers[-1].out_type
                layers.append(escnn_nn.PointwiseAvgPool2D(in_type, kernel_size=2, stride=2, padding=0, ceil_mode=True))
                # in_type comes from the out_type of previous layers
            # first layer
            else:
                # i = 0
                # we store the input type for wrapping the images into a geometric tensor during the forward pass
                # We need to mask the input image since the corners are moved outside the grid under rotations
                
                layers.append(escnn_nn.MaskModule(self.input_type, 128, margin=1))
            
            layers.append(escnn_nn.R2Conv(in_type, out_type, kernel_size=3, padding=int(padding), bias = False))
            layers.append(escnn_nn.InnerBatchNorm(out_type))
            layers.append(escnn_nn.ReLU(out_type, inplace=True))

            #for _ in range(no_convs_per_block-1):
            #    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
            #    layers.append(nn.ReLU(inplace=True))


        # aggregate the layer and make a model
        # aggregate group operations
        layers.append(escnn_nn.GroupPooling(out_type))
        self.layers = escnn_nn.SequentialModule(*layers)
        # init
        # self.layers_apply(init_weights)


    def forward(self, input):
        # convert into geometric tensor
        input_geom = escnn_nn.GeometricTensor(input, self.input_type)
        #input_geom = self.input_type(input)
        input_geom = self.layers(input_geom)
        return input_geom.tensor

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist

"""
NN to return Von Mises Fisher distributions on the Kendall Shape space instead of vanilla Gaussian.
Return mu, concentration, and rotation vector
"""
class KendallShapeVmf(AxisAlignedConvGaussian):

    def __init__(self, input_channels, num_filters, no_convs_per_block, k, m, initializers, posterior=False):
        super().__init__(input_channels, num_filters, no_convs_per_block, k * m, initializers, posterior)
        # append last filter as latent dim
        self.k = k
        self.m = m
        self.encoder_mu = EquiEncoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        # out_dim = self.encoder_mu.layers[-1].out_type.size
        self.conv_layer_mu = nn.Conv2d(num_filters[-1], self.latent_dim, kernel_size = 1, stride = 1)
        # append last filter as 1 for concent
        self.encoder_concent_rot = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior = self.posterior)
        #self.layer_concent = nn.Linear(self.num_filters[-1], 1)
        #self.layer_rot = nn.Linear(self.num_filters[-1], 2)
        # append last filter as 2 for rotation matrix
        # self.encoder_rot = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior = self.posterior)
        self.conv_layer_concent = nn.Conv2d(num_filters[-1], 1, kernel_size = (1,1), stride = 1)
        # return SO(m); need not be rotation equivariant
        self.conv_layer_rot = nn.Conv2d(num_filters[-1], 2, kernel_size = (1,1), stride = 1)


    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)
        # encoder returns mu
        encoding_mu = self.encoder_mu(input)
        self.show_enc = encoding_mu
        #We only want the mean of the resulting hxw image
        encoding_mu = torch.mean(encoding_mu, dim = 2, keepdim = True)
        encoding_mu = torch.mean(encoding_mu, dim = 3, keepdim = True)
        mu = self.conv_layer_mu(encoding_mu)
        # we squeeze the second dimension twice since o.w. won't work when batch size == 1
        mu = torch.squeeze(mu, dim = 2)
        mu = torch.squeeze(mu, dim = 2)
        # m = 2 k = 4 for now.
        # resize to the pre-shape space matrix
        mu = mu.view(-1, self.m, self.k)
        # m = 2 k = 4 for now. vmf loc parameter
        # mean 0, unit vector columns
        mu_mean = torch.mean(mu, dim = 1)
        # mean 0
        torch.sub(mu, mu_mean[:, None])
        # normalize
        mu = mu / mu.norm(p = 2.0)
        # other variables need not be rotation invariant
        concent_rot = self.encoder_concent_rot(input)
        concent_rot = torch.mean(concent_rot, dim = 2, keepdim = True)
        concent_rot = torch.mean(concent_rot, dim = 3, keepdim = True)
        concent= self.conv_layer_concent(concent_rot)
        rot = self.conv_layer_rot(concent_rot)
        concent = torch.squeeze(concent, dim = 2)
        concent = torch.squeeze(concent, dim = 2)
        rot = torch.squeeze(rot, dim = 2)
        rot = torch.squeeze(rot, dim = 2)
        # + 1 prevent collapsing behavior 
        concent = F.softplus(concent) + 1
        # rotation matrix
        rot = F.normalize(rot, p = 2.0, dim = 1)
        # make rotation matrices
        print(concent.shape)
        rot = torch.stack((rot[:, 0], -rot[:, 1], rot[:, 1], rot[:, 0])).T.view(-1, self.m, self.m)
        print(rot.shape)
        # rot = torch.tensor([[rot[0], -rot[1]],[rot[1], rot[0]]])

        # TODO: Implement rotation invariant vmf distribution, by mu_0 = rot^-1 *  mu
        mu = torch.linalg.solve(rot, mu)

        # vmf distribution with parameters from NN, with rotation invariance.
        # for scaling and translation invariance, you first center and scale the input data.
        # dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        # convert 2 * 4 matrix by S^((k - 1) * m - 1)
        dist = VonMisesFisher(mu, concent)
        return dist


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)

class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch,False)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            #z_prior = self.prior_latent_space.base_dist.loc 
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)
        z_posterior = self.posterior_latent_space.rsample()
        
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)
        
        reconstruction_loss = criterion(input=self.reconstruction, target=segm)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + self.beta * self.kl)

"""
Probabilistic Unet with Kendall Shape space embedding
"""
class KendallProbUnet(ProbabilisticUnet):


    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], k = 4, m = 2, no_convs_fcomb=4, beta=10.0):
        super().__init__(input_channels, num_classes, num_filters, k * m, no_convs_fcomb, beta)
        self.prior = KendallShapeVmf(self.input_channels, self.num_filters, self.no_convs_per_block, k, m,  self.initializers).to(device)
        self.posterior = KendallShapeVmf(self.input_channels, self.num_filters, self.no_convs_per_block, k, m, self.initializers, posterior=True).to(device)
