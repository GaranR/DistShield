import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def draw(weights, name):
    plt.figure(figsize=(10, 6))
    bin_width = 0.005
    bins = np.arange(-0.8, 0.8, bin_width)
    plt.hist(weights, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.hist(weights[weights > 0.1], bins=bins, alpha=0.75, color='red', edgecolor='black')
    plt.title(f'{name}')
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.xlim(-0.8, 0.8)
    # plt.ylim(0, 6000)
    plt.ylim(0, 200)
    plt.grid(True)
    plt.show()
    return


def plot_weight_change_with_histogram(weights_pre, weights_post, name, lim):
    bin_width = 0.005
    bins = np.arange(-0.25, 0.25, bin_width)
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.hist(weights_post.flatten(), bins=bins, alpha=0.5, color='green', edgecolor='black', label='NNSplitter')
    plt.hist(weights_pre.flatten(), bins=bins, alpha=0.5, color='blue', edgecolor='black', label='Ori_Model')
    weight_diff = weights_post - weights_pre
    changed_indices = np.where(np.abs(weight_diff) > 1e-5)
    y_offset = 0.05
    for idx in zip(*changed_indices):
        pre_value = weights_pre[idx]
        post_value = weights_post[idx]
        pre_bin_idx = np.digitize(pre_value, bins) - 1
        post_bin_idx = np.digitize(post_value, bins) - 1
        pre_bin_center = (bins[pre_bin_idx] + bins[pre_bin_idx + 1]) / 2
        post_bin_center = (bins[post_bin_idx] + bins[post_bin_idx + 1]) / 2
        plt.annotate(
            '',
            xy=(post_bin_center, y_offset),
            xytext=(pre_bin_center, y_offset),
            arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=1)  # 改细箭头
        )
        y_offset += 1

    plt.axvline(x=-0.145, color='red', linestyle='--', label='Weight Limit')
    plt.axvline(x=0.24, color='red', linestyle='--', label='')
    plt.title(f'Weight Change from {name} (Origin to NNSplitter)', fontsize=16)
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.xlim(-0.2, 0.3)
    plt.ylim(0, lim)
    plt.legend(loc='upper right')
    plt.show()


def plot_new(weights_pre, weights_post, name, lim, ax):
    bin_width = 0.0075
    bins = np.arange(-0.2, 0.3, bin_width)
    pre_freq, pre_bins = np.histogram(weights_pre.flatten(), bins=bins)
    ax.bar(pre_bins[:-1], pre_freq, width=bin_width, align='edge', alpha=0.3, color='lightblue', edgecolor='black',
           label='Original Weights')
    post_freq, post_bins = np.histogram(weights_post.flatten(), bins=bins)
    ax.bar(post_bins[:-1], -post_freq, width=bin_width, align='edge', alpha=0.2, color='red', edgecolor='black',
           label='NNSplitter')
    weight_diff = weights_post - weights_pre
    changed_indices = np.where(np.abs(weight_diff) > 1e-5)
    pre_values = weights_pre[changed_indices]
    post_values = weights_post[changed_indices]
    pre_freq_changed, pre_bins_changed = np.histogram(pre_values.flatten(), bins=bins)
    post_freq_changed, post_bins_changed = np.histogram(post_values.flatten(), bins=bins)
    ax.bar(pre_bins_changed[:-1], pre_freq_changed, width=bin_width, align='edge', alpha=0.6, color='blue',
           edgecolor='black', label='Weights Before Changed')
    ax.bar(post_bins_changed[:-1], -post_freq_changed, width=bin_width, align='edge', alpha=0.6, color='red',
           edgecolor='black', label='Weights After Changed')
    ax.set_xlim(-0.2, 0.3)
    ax.set_ylim(-lim, lim)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(f'{name}', fontsize=16)
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Frequency')
    ax.axhline(0, color="k", linewidth=1)
    ax.axvline(x=-0.145, color='gray', linestyle='--', label='weight limit')
    ax.axvline(x=0.2425, color='gray', linestyle='--')
    ax.legend(loc='upper right')
    def abs_formatter(x, pos):
        return f'{abs(x):.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(abs_formatter))


def compare_weights_and_plot(net, net_pre, layer_name_substr, lim, ax):
    weights_pre_all = []
    weights_post_all = []
    if layer_name_substr == 'All Conv Layers':
        for name, module in net_pre.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                weights_pre_all.append(module.weight.data.cpu().numpy().flatten())
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                weights_post_all.append(module.weight.data.cpu().numpy().flatten())
    else:
        for name, module in net_pre.named_modules():
            if isinstance(module, torch.nn.Conv2d) and layer_name_substr in name:
                weights_pre_all.append(module.weight.data.cpu().numpy().flatten())
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) and layer_name_substr in name:
                weights_post_all.append(module.weight.data.cpu().numpy().flatten())

    if weights_pre_all and weights_post_all:
        weights_pre_all = np.concatenate(weights_pre_all, axis=0)
        weights_post_all = np.concatenate(weights_post_all, axis=0)
        plot_new(weights_pre_all, weights_post_all, f'{layer_name_substr} Weight Change', lim, ax)
    else:
        print(f"No convolutional layers matching '{layer_name_substr}' found in one or both models.")


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

def inference(net, device, testloader):
    net.to(device)
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct / total
    return acc


def js_div(p_out, q_out, get_softmax=True):
    kld_div_loss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_out = torch.softmax(p_out, dim=-1)
        q_out = torch.softmax(q_out, dim=-1)
    log_mean_out = ((p_out + q_out) / 2).log()
    return (kld_div_loss(log_mean_out, p_out) + kld_div_loss(log_mean_out, q_out)) / 2
