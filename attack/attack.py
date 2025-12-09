import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import test_outlier
from utils import *
from datasets import load_dataset
# import your model here
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = 3
    if dataset == 0:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)
        # original model
        net_pre = resnet18(pretrained=True)
        # obfuscated model
        net = resnet18()
        pth_file_path = 'my_resnet_mobile_op.pth'
        net.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
    else:
        print("load other dataset and models")

    all_weight = []
    pre_max = {}
    pre_min = {}
    pre = {}
    overflow = 0
    detected_overflow = 0
    detect = 0
    for name, module in net_pre.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pre[name] = module.weight.data.view(-1)

    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name)
            weights = module.weight.data
            flattened_weights = weights.view(-1)
            outlier_indices = test_outlier.detect_outliers_with_mask(name, flattened_weights, bin_width=0.03, iterations=2, threshold=0, ratio=0.001)
            detected_overflow_t = len(outlier_indices)
            detected_overflow += detected_overflow_t
            weight_diff = flattened_weights - pre[name]
            changed_indices = np.where(np.abs(weight_diff) > 1e-5)[0]
            overflow_t = changed_indices.size
            overflow += overflow_t
            overlap_indices = np.intersect1d(outlier_indices, changed_indices)
            detect_t = len(overlap_indices)
            print(f"Detected overflow successfully: {detect_t}/{overflow_t},{detected_overflow_t}")
            detect += detect_t
            flattened_weights[outlier_indices] = 0
            weights[:] = flattened_weights.view(weights.size())

    acc = inference(net, device, test_loader)
    print(f'accuracy: {acc}')
    print(f'all: {overflow}')
    print(f'detected: {detect}')
    print(f'ratio: {detect/overflow}')