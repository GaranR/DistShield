import numpy as np
import torch.backends.cudnn as cudnn
import torch
import random
import argparse
import logging
import time
import os
import sys
from train import *
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from utils import *
# import your model here
from PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18, resnet34
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2

from datasets import load_dataset




class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]  # PIL Image
        label = example["label"]  # Integer label

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: Could not find GPU! Use CPU only")

    dataset = 0
    net_name = 2
    ratio = 1 / 100

    if dataset == 0:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
        trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    elif dataset == 1:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])
        trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    elif dataset == 2:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
        trainset = torchvision.datasets.STL10(root='./', split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.STL10(root='./', split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    elif dataset == 3:
        ds = load_dataset("timm/resisc45")
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_dataset = HuggingFaceDataset(ds["train"], transform=transform)
        test_dataset = HuggingFaceDataset(ds["test"], transform=transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)

    net = resnet18(pretrained=True)
    # net set to you ori model
    net.to(device)
    print('start')
    cnt = 0
    for name, module in net.named_modules():
        name = name + '.weight'
        if isinstance(module, torch.nn.Conv2d):
            if net_name == 1 and 'conv' not in name:
                continue
            weights = module.weight.data
            flattened_weights = weights.view(-1)
            sorted = torch.argsort(flattened_weights)
            replace_count = int(ratio * len(flattened_weights))
            if replace_count == 0:
                continue
            cnt += replace_count
            max_idx = sorted[-replace_count:]
            mask = torch.ones_like(flattened_weights, dtype=torch.bool)
            mask[max_idx] = False
            mean = torch.mean(flattened_weights[mask], dim=0).item()
            std = torch.std(flattened_weights[mask], dim=0).item()
            random_replacement = torch.normal(mean, std, size=(replace_count, )).to(device)
            flattened_weights[max_idx] = random_replacement
    print(f'{cnt} weights changed')
    acc = inference(net, device, testloader)
    print(f'accuracy: {acc}')
    torch.save(net.state_dict(), f'./result/mag_mob_c10.pth')

if __name__ == '__main__':
    main()
