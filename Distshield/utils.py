

import torch
import torch.nn as nn


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
