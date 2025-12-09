# TODO: 1) locate weights; 2) check the magnitude of modified weights

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import *
# import your model here
from PyTorch_CIFAR10.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18, resnet34
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2



def compute_gradient_magnitude_per_layer(model, criterion, data_loader, device):
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.to(device)
    model.to(device)
    model.eval()
    output = model(data)
    loss = criterion(output, target)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if m.weight.grad is not None:
                m.weight.grad.data.zero_()
    loss.backward()
    return


def draw(weights, name, mask):
    plt.figure(figsize=(10, 6))
    bin_width = 0.005
    bins = np.arange(-0.2, 0.2, bin_width)
    plt.hist(weights.view(-1).cpu(), bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.hist(weights[mask[name + '.weight'] > 0].view(-1).cpu(), bins=bins, alpha=0.75, color='red', edgecolor='black')
    plt.title(name)
    plt.xlim([-0.4, 0.4])
    plt.ylim([0, 500])
    plt.grid(True)
    # plt.show()
    plt.savefig(f"imgs/{name}.png")
    plt.close()
    return


def Trainer(arg, net_name, net, trainloader, testloader, device, my_op):
    net_ori = resnet18(pretrained=True)
    # net_ori set to the copy of you ori model
    net_ori.to(device)
    ori_dict = {}
    ori_max = {}
    ori_min = {}
    for name, module in net_ori.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            ori_dict[name + '.weight'] = module.weight
            flattened_para = module.weight.view(-1)
            sorted_tensor, _ = torch.sort(flattened_para)
            num_elements = flattened_para.numel()
            top_1_percent_idx = num_elements - 50
            bottom_1_percent_idx = 50
            top_1_percent_value = sorted_tensor[top_1_percent_idx].item()
            bottom_1_percent_value = sorted_tensor[bottom_1_percent_idx].item()
            ori_max[name + '.weight'] = top_1_percent_value
            ori_min[name + '.weight'] = bottom_1_percent_value
    net_tmp = resnet18(pretrained=True)
    # net_tmp set to the copy of you ori model
    net_tmp.to(device)
    val_accuracy = 100.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=arg.lr)
    ratio = arg.ratio
    mask = {}
    cnt_w = 0
    idx_list = []
    ori_w = []
    compute_gradient_magnitude_per_layer(net_tmp, criterion, trainloader, device)
    for name, module in net_tmp.named_modules():
        name = name + '.weight'
        if isinstance(module, torch.nn.Conv2d):
            if net_name == 1 and 'conv' not in name:
                continue
            weights = module.weight.data
            flattened_weights = weights.view(-1)
            replace_count = int(ratio * len(flattened_weights))
            replace_count = max(0, replace_count)
            cnt_w += replace_count

            if arg.random == 1:
                random_idx = torch.randperm(flattened_weights.size(0))[:replace_count]
                mask[name] = torch.zeros_like(flattened_weights)
                mask[name][random_idx] = 1
            else:
                w_grad_topk, w_idx_topk = module.weight.grad.detach().abs().view(-1).topk(replace_count)
                mask[name] = torch.zeros_like(flattened_weights)
                mask[name][w_idx_topk] = 1

            mask[name] = mask[name].view(weights.size())
            indexes = torch.nonzero(mask[name], as_tuple=True)
            idx_list.append(indexes)
            w = module.weight.data[indexes]
            ori_w.extend(w.cpu().detach().numpy())
    print(mask.keys())
    last_temp = 0
    for epoch in range(arg.num_epoch):
        net.eval()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss_c = criterion(outputs, targets)
            cnt = 0
            kld = 0
            for name, para in net.named_parameters():
                if name in mask:
                    flattened_para = para[mask[name] > 0].view(-1)
                    flattened_ori = ori_dict[name][mask[name] > 0].view(-1)
                    flattened_para = flattened_para[torch.argsort(flattened_para)]
                    flattened_ori = flattened_ori[torch.argsort(flattened_ori)]
                    cnt += 1
                    if flattened_para.numel() != 0:
                        kl_t1 = torch.log_softmax(flattened_para, dim=0)
                        kl_t2 = torch.softmax(flattened_ori, dim=0)
                        kld_tmp = F.kl_div(kl_t1, kl_t2, reduction='sum')
                        if kld_tmp.item() > 0.0:
                            kld += kld_tmp

            if arg.dataset == 0:
                if net_name == 0:                       # vgg
                    loss = -loss_c + 2 ** (kld * 2e5)
                if net_name == 1:                       # resnet
                    loss = -loss_c + 2 ** (kld * 5e3)
                if net_name == 2:                       # mobilenet
                    loss = -loss_c + 2 ** (kld * 5e4)
            elif arg.dataset == 1:
                if net_name == 0:                       # vgg
                    loss = -loss_c + 2 ** (kld * 1.5e4)
                if net_name == 1:                       # resnet
                    loss = -loss_c + 2 ** (kld * 3.5e3)
                if net_name == 2:                       # mobilenet
                    loss = -loss_c + 2 ** (kld * 5e2)
            elif arg.dataset == 2:
                if net_name == 0:                       # vgg
                    loss = -loss_c + 2 ** (kld * 3e4)
                if net_name == 1:                       # resnet
                    loss = -loss_c + 2 ** (kld * 8e3)
                if net_name == 2:                       # mobilenet
                    loss = -loss_c + 2 ** (kld * 6.5e2)
            elif arg.dataset == 3:
                if net_name == 0:                       # vgg
                    loss = -loss_c + 2 ** (kld * 2.1e3)
                if net_name == 1:                       # resnet
                    loss = -loss_c + 2 ** (kld * 4e3)
                if net_name == 2:                       # mobilenet
                    loss = -loss_c + 2 ** (kld * 4e2)

            loss.backward()

            for name, para in net.named_parameters():
                if name in mask:
                    para.grad *= mask[name].long()
                else:
                    para.grad *= 0

            optimizer.step()

            with torch.no_grad():
                for name, para in net.named_parameters():
                    if name in mask:
                        para.data = (1 - mask[name]) * para + (mask[name] * para).clamp_(ori_min[name] * 0.85,
                                                                                         ori_max[name] * 0.75)
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum()


        min_w = []
        max_w = []
        cnt_w_re = 0
        layer_modi_re = []
        cos_sim = 0
        cnt = 0
        jsd = 0
        kld = 0
        total_cnt = 0
        for name, para in net.named_parameters():
            if len(para.shape) > 1 and name in mask:
                if net_name != 2 and para.shape[-1] <= 1:
                    continue
                tmp = para * mask[name]
                min_w_tmp = torch.min(tmp)
                max_w_tmp = torch.max(tmp)
                min_w.append(min_w_tmp)
                max_w.append(max_w_tmp)
                modi_w = mask[name] > 0
                num_modi_re = modi_w.sum()
                cnt_w_re += num_modi_re
                layer_modi_re.append(num_modi_re.item())
                flattened_para = para[mask[name] > 0].view(-1)
                flattened_ori = ori_dict[name][mask[name] > 0].view(-1)
                flattened_para = flattened_para[torch.argsort(flattened_para)]
                flattened_ori = flattened_ori[torch.argsort(flattened_ori)]
                cos_sim += torch.cosine_similarity(flattened_para, flattened_ori, dim=0).item()
                total_cnt += ori_dict[name].numel()
                kld += F.kl_div(torch.log_softmax(flattened_para, dim=0), torch.softmax(flattened_ori, dim=0), reduction='sum')
                cnt += 1

        min_w_v = min(min_w)
        max_w_v = max(max_w)
        kld = 2 ** (kld * 2e3)
        print('============cos: %.5f' % (cos_sim / cnt))
        print('============kld: %f' % kld)
        print(
            '==========================[epoch:%d] Loss: %.03f | Acc: %.3f%% | ori_num:%d \n| cur_num:%d'
            % (epoch + 1, loss.item(), 100. * correct / total, cnt_w, cnt_w_re))

        print('==========================min: %.5f | max: %.5f '
              % (min_w_v, max_w_v))

        acc = inference(net, device, testloader)
        print(f'Val: | Acc: {acc:.5f}')

        # for name, module in net.named_modules():
        #     if net_name == 1:
        #         if isinstance(module, torch.nn.Conv2d) and 'conv' in name:
        #             draw(module.weight.data, name, mask)
        #     elif net_name == 0:
        #         if isinstance(module, torch.nn.Conv2d):
        #             draw(module.weight.data, name, mask)
        #     elif net_name == 2:
        #         if isinstance(module, torch.nn.Conv2d):
        #             draw(module.weight.data, name, mask)

        if my_op == 1:
            if arg.dataset == 0 and acc > 0.105:
                continue
            elif arg.dataset == 1 and acc > 0.015:
                continue
            elif arg.dataset == 2 and acc > 0.105:
                continue
            elif arg.dataset == 3 and acc > 0.03:
                continue
            torch.save(net.state_dict(), arg.PATH)
            last_temp = cnt_w_re
            step = 0.1
            with torch.no_grad():
                for name, module in net.named_modules():
                    name = name + '.weight'
                    if isinstance(module, torch.nn.Conv2d):
                        if net_name == 1 and 'conv' not in name:
                            continue
                        weights = module.weight.data
                        replace_count = int(step * mask[name].sum())
                        if replace_count == 0:
                            replace_count = 1
                        temp_weight = module.weight.grad.detach().abs() * mask[name]
                        temp_weight += 1 - mask[name]
                        temp_weight = temp_weight.view(-1)
                        _, w_idx_mink = temp_weight.topk(replace_count, largest=False)
                        zero_mask = torch.zeros_like(temp_weight)
                        zero_mask[w_idx_mink] = 1
                        mask[name][zero_mask.view(mask[name].size()) > 0] = 0
                        weights[zero_mask.view(mask[name].size()) > 0] = ori_dict[name][zero_mask.view(mask[name].size()) > 0]
            optimizer = optim.Adam(net.parameters(), lr=arg.lr)

    np.save('mask.npy', mask)
    print(f'==============last temp: {last_temp}')
    return val_accuracy, net.state_dict()
