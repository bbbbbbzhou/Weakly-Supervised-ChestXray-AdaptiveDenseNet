import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks_classify import DenseNet, DenseNetADA
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts):
    if name == 'densenetADA':
        network = DenseNetADA(in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                              bn_size=4, drop_rate=0, num_classes=opts.n_class,
                              num_maps=opts.n_maps, kmax=opts.kmax, kmin=opts.kmin, alpha=opts.alpha)

    elif name == 'densenet':
        network = DenseNet(in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                           bn_size=4, drop_rate=0, num_classes=opts.n_class)

    else:
        raise NotImplementedError

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)
