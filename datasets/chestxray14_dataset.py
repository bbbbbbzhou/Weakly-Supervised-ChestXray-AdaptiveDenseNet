import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image


class ChestXray14_Train(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'images')
        self.label_file = os.path.join(self.root, 'labels', 'train_list.txt')

        image_names = []
        labels = []
        with open(self.label_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(self.data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_names = image_names
        self.labels = labels

        self.n_classes = 14
        self.class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass'
                           'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
                           'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        self.AUG = opts.AUG
        self.T_train = transforms.Compose([
            transforms.Scale(opts.train_osize, Image.BICUBIC),
            transforms.RandomRotation(opts.train_angle, fill=(0,)),
            transforms.RandomCrop(opts.train_fineSize),
            transforms.RandomHorizontalFlip(),
        ])
        self.T_must = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('L')
        label = self.labels[index]
        if self.AUG:
            image = self.T_train(image)
        image = self.T_must(image)

        return {'img': image, 'label': torch.FloatTensor(label)}

    def __len__(self):
        return len(self.image_names)


class ChestXray14_Vali(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'images')
        self.label_file = os.path.join(self.root, 'labels', 'val_list.txt')

        image_names = []
        labels = []
        with open(self.label_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(self.data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_names = image_names
        self.labels = labels

        self.n_classes = 14
        self.class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass'
                           'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
                           'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        self.T_must = transforms.Compose([
            transforms.Scale(opts.eval_osize, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('L')
        label = self.labels[index]

        image = self.T_must(image)

        return {'img': image, 'label': torch.FloatTensor(label)}

    def __len__(self):
        return len(self.image_names)


class ChestXray14_Test(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.data_dir = os.path.join(self.root, 'images')
        self.label_file = os.path.join(self.root, 'labels', 'test_list.txt')

        image_names = []
        labels = []
        with open(self.label_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(self.data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        self.image_names = image_names
        self.labels = labels

        self.n_classes = 14
        self.class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass'
                                                                                      'Nodule', 'Pneumonia',
                           'Pneumothorax', 'Consolidation', 'Edema'
                                                            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

        self.T_must = transforms.Compose([
            transforms.Scale(opts.eval_osize, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('L')
        label = self.labels[index]

        image = self.T_must(image)

        return {'img': image, 'label': torch.FloatTensor(label)}

    def __len__(self):
        return len(self.image_names)


if __name__ == '__main__':
    pass
