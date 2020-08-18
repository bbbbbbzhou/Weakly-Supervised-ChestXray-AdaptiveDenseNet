import os
import numpy as np
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import entr
import pdb

from networks import get_generator
from networks.networks_classify import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, mse, get_nonlinearity
from skimage.measure import compare_ssim as ssim


class CNNModel(nn.Module):
    def __init__(self, opts):
        super(CNNModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        # set default loss flags
        loss_flags = ("w_img_BCE")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        if self.is_train:
            self.loss_names += ['loss_G_BCE']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.BCEWithLogitsLoss()

        self.opts = opts

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.img = data['img'].to(self.device).float()
        self.labels_gt = data['label'].to(self.device).float()

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        inp = self.img
        inp.requires_grad_(True)
        self.labels_pred, self.heatmaps = self.net_G(inp)

    def update_G(self):
        loss_G_BCE = 0
        self.optimizer_G.zero_grad()
        loss_G_BCE = self.criterion(self.labels_pred, self.labels_gt)
        self.loss_G_BCE = loss_G_BCE.item()

        total_loss = loss_G_BCE
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_BCE = 0
        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        message += 'G_BCE: {:.4e} '.format(self.loss_G_BCE)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        state['net_G'] = self.net_G.module.state_dict()
        state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location='cuda:0')

        self.net_G.module.load_state_dict(checkpoint['net_G'])
        if train:
            self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)

        labels_pred_all = []
        labels_gt_all = []
        heatmap_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            labels_pred_all.append(self.labels_pred[0])
            labels_gt_all.append(self.labels_gt[0])

            heatmap_images.append(self.heatmaps[0].cpu())

        labels_pred_all = torch.stack(labels_pred_all).squeeze().cpu().numpy()
        labels_gt_all = torch.stack(labels_gt_all).squeeze().cpu().numpy()
        all_Ap, mAp = metric_mAp(labels_pred_all, labels_gt_all)
        all_AUC, mAUC = metric_mAUC(labels_pred_all, labels_gt_all)

        self.all_Ap = all_Ap
        self.mAp = mAp
        self.all_AUC = all_AUC
        self.mAUC = mAUC

        n_class = labels_gt_all.shape[1]
        message = ''
        for i in range(n_class):
            message += 'Ap_class{}: {:4f} '.format(i, all_Ap[i])
        message += 'mAp: {:4f} '.format(mAp)
        print(message)

        message = ''
        for i in range(n_class):
            message += 'AUC_class{}: {:4f} '.format(i, all_AUC[i])
        message += 'mAUC: {:4f} '.format(mAUC)
        print(message)

        self.results = {}
        self.results['labels_pred_all'] = labels_pred_all
        self.results['labels_gt_all'] = labels_gt_all
        self.results['heatmaps'] = torch.stack(heatmap_images).squeeze().numpy()


def metric_mAp(output, target):
    """ Calculation of mAp """
    output_np = output
    target_np = target

    num_class = target.shape[1]
    all_ap = []
    for cid in range(num_class):
        gt_cls = target_np[:, cid].astype('float32')
        pred_cls = output_np[:, cid].astype('float32')

        TP = np.sum(gt_cls * pred_cls)
        FP = np.sum(gt_cls * (1 - pred_cls))

        if TP == 0 and FP == 0:
            continue
        else:
            pred_cls = pred_cls - 1e-5 * gt_cls
            ap = average_precision_score(gt_cls, pred_cls, average=None)

        all_ap.append(ap)

    mAP = np.mean(all_ap)
    return all_ap, mAP


def metric_mAUC(output, target):
    """ Calculation of ROC AUC """
    output_np = output
    target_np = target

    num_class = target.shape[1]
    all_roc_auc = []
    for cid in range(num_class):
        gt_cls = target_np[:, cid].astype('float32')
        pred_cls = output_np[:, cid].astype('float32')

        TP = np.sum(gt_cls * pred_cls)
        FP = np.sum(gt_cls * (1 - pred_cls))

        if TP == 0 and FP == 0:
            continue
        else:
            pred_cls = pred_cls - 1e-5 * gt_cls
            roc_auc = roc_auc_score(gt_cls, pred_cls, average='weighted')

        all_roc_auc.append(roc_auc)

    mROC_AUC = np.mean(all_roc_auc)
    return all_roc_auc, mROC_AUC