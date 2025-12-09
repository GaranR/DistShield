import numpy as np
import torch
import argparse
from train import *
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from datasets import load_dataset
# import your model here
from PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18, resnet34
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2



parser = argparse.ArgumentParser('model')
parser.add_argument('--model', type=int, default=2) # 0 - vgg11_bn; 1 - resnet18; 2 - mobilenetv2
parser.add_argument('--dataset', type=int, default=3) # 0 - cifar10; 1 - cifar100; 2 - stl10; 3 - resisc45
parser.add_argument('--lr', type=float, default=0.005) # 0.005
parser.add_argument('--num_epoch', type=int, default=20) # 20
parser.add_argument('--PATH', type=str, default='./result/my_temp.pth')
parser.add_argument('--random', type=int, default=0)
parser.add_argument('--opti', type=int, default=1)
parser.add_argument('--ratio', type=int, default=0.001)
args = parser.parse_args()


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

    # %% ========= Get data ==========
    if args.dataset == 0:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
        trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    elif args.dataset == 1:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])
        trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR100(root='./', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    elif args.dataset == 2:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])
        trainset = torchvision.datasets.STL10(root='./', split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.STL10(root='./', split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    elif args.dataset == 3:
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
    Trainer(args, args.net, net, trainloader, testloader, device, my_op=args.opti)
    # torch.save(net_dict_new, args.PATH)


if __name__ == '__main__':
    main()
