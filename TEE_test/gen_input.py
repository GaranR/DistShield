import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PyTorch_CIFAR10_master.cifar10_models.resnet import resnet18
from PyTorch_CIFAR10_master.cifar10_models.vgg import vgg11_bn
from PyTorch_CIFAR10_master.cifar10_models.mobilenetv2 import mobilenet_v2
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

model = resnet18(pretrained=True)
# model = vgg11_bn(pretrained=True)
# model = mobilenet_v2(pretrained=True)
model.eval()
device = torch.device("cpu")
model = model.to(device)

layer_inputs = {}
layer_outputs = {}
layer_weights = {}


def extract_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight
            weight_flat = weight.view(weight.size(0), -1)
            layer_weights[name] = weight.shape


def hook_fn(module, input, output):
    unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)
    unfolded_input = unfold(input[0])
    print(unfolded_input[0].shape)
    layer_inputs[module.name] = unfolded_input[0]


def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and 'conv' in name:
            module.name = name
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    return hooks

if __name__ == '__main__':
    extract_weights(model)
    print(layer_weights)
    print()
    print()
    hooks = register_hooks(model)
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(test_loader):
            if i == 0:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                print(f'Ground truth label: {labels[0]}')
                print(f'Predicted label: {predicted[0]}')
            break

    for layer_name, input_matrix in layer_inputs.items():
        file_name = f'layers/{layer_name}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(input_matrix.cpu().detach().numpy(), f)
