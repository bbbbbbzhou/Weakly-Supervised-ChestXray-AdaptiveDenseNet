import os
import argparse
import json
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model
import scipy.io as sio
import csv
import pdb

parser = argparse.ArgumentParser(description='Weakly Supervised Learning for Chest X-ray')

# model name
parser.add_argument('--experiment_name', type=str, default='train_ChestXray14_densenetADA', help='give a model name before training')
parser.add_argument('--model_type', type=str, default='model_wsl', help='type of model: model_wsl')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--dataset', type=str, default='ChestXray14', help='dataset name')
parser.add_argument('--data_root', type=str, default='../Data/ChestXray14/', help='data root folder')

# network
parser.add_argument('--net_G', type=str, default='densenetADA', help='densenet / densenetADA')
parser.add_argument('--n_class', type=int, default=14, help='number of class type in classification')

# wildcat options
parser.add_argument('--n_maps', type=int, default=3, help='number of maps for class-wise pooling')
parser.add_argument('--kmax', type=float, default=1, help='kmax for spatial pooling')
parser.add_argument('--kmin', type=float, default=None, help='kmin for spatial pooling')
parser.add_argument('--alpha', type=float, default=1, help='alpha for spatial pooling')

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--AUG', default=False, action='store_true', help='use augmentation')
parser.add_argument('--train_osize', type=int, default=270, help='random scale')
parser.add_argument('--train_angle', type=int, default=20, help='random rotation angle')
parser.add_argument('--train_fineSize', nargs='+', type=float, default=[256, 256], help='random crop')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=4, help='save evaluation for every number of epochs')
parser.add_argument('--eval_osize', type=int, default=256, help='random scale')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=4, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters: {} \n'.format(num_param))

if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

# select dataset
train_set, val_set, test_set = get_datasets(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# evaluation

print('Test Evaluation ......')
model.eval()
with torch.no_grad():
    model.evaluate(test_loader)
sio.savemat(os.path.join(image_directory, 'eval.mat'), model.results)
