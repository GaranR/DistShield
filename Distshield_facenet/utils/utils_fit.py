import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import evaluate
import matplotlib.pyplot as plt
import torch.optim as optim


def draw(weights, name, mask):
    plt.figure(figsize=(10, 6))
    bin_width = 0.01
    bins = np.arange(-0.7, 0.7, bin_width)
    plt.hist(weights.view(-1).cpu(), bins=bins, alpha=0.75, color='blue', edgecolor='black')
    plt.hist(weights[mask[name + '.weight'] > 0].view(-1).cpu(), bins=bins, alpha=0.75, color='red', edgecolor='black')
    plt.title(name)
    plt.xlim([-0.7, 0.7])
    plt.ylim([0, 500])
    plt.grid(True)
    # plt.show()
    plt.savefig(f"imgs/{name}.png")
    plt.close()
    return

def compute_gradient_magnitude_per_layer(model, loss, gen, cuda, local_rank, optimizer, Batch_size):
    for iteration, batch in enumerate(gen):
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                labels  = labels.cuda(local_rank)

        optimizer.zero_grad()
        outputs1, outputs2 = model(images, "train")

        _triplet_loss   = loss(outputs1, Batch_size)
        _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
        _loss           = _triplet_loss + _CE_loss
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        _loss.backward()
        break
    return

def fit_one_epoch(Init_lr_fit, Init_Epoch, net_ori, net_tmp, model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, test_loader, Batch_size, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank, re_op):
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    ori_dict = {}
    ori_max = {}
    ori_min = {}
    for name, module in net_ori.cuda().named_modules():
        if isinstance(module, torch.nn.Conv2d):
            ori_dict['module.' + name + '.weight'] = module.weight
            flattened_para = module.weight.view(-1)
            sorted_tensor, _ = torch.sort(flattened_para)
            num_elements = flattened_para.numel()
            top_1_percent_idx = num_elements - 50
            bottom_1_percent_idx = 50
            top_1_percent_value = sorted_tensor[top_1_percent_idx].item()
            bottom_1_percent_value = sorted_tensor[bottom_1_percent_idx].item()
            ori_max['module.' + name + '.weight'] = top_1_percent_value
            ori_min['module.' + name + '.weight'] = bottom_1_percent_value
    compute_gradient_magnitude_per_layer(net_tmp.cuda(), loss, gen, cuda, local_rank, optimizer, Batch_size)
    ratio = 0.0009
    random = 0
    mask = {}
    cnt_w = 0
    idx_list = []
    ori_w = []
    for name, module in net_tmp.named_modules():
        name = 'module.' + name + '.weight'
        if isinstance(module, torch.nn.Conv2d):
            if 'backbone' not in name:
                continue
            if 'branch0' in name or '3.2.branch1.2' in name or '3.3.branch1.2' in name:
                continue
            weights = module.weight.data
            flattened_weights = weights.view(-1)
            replace_count = int(ratio * len(flattened_weights))
            replace_count = max(250, replace_count)
            if '3.4.branch1.2' in name or '3.3.conv2d' in name or '3.0.branch1.2' in name:
                replace_count = 100
            cnt_w += replace_count

            # random
            if random == 1:
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

    for epoch in range(Init_Epoch, Epoch):
        total_triple_loss   = 0
        total_CE_loss       = 0
        total_accuracy      = 0

        val_total_triple_loss   = 0
        val_total_CE_loss       = 0
        val_total_accuracy      = 0

        if local_rank == 0:
            print('Start Train')
            pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        model_train.train()
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images  = images.cuda(local_rank)
                    labels  = labels.cuda(local_rank)

            optimizer.zero_grad()
            cnt = 0
            kld = 0
            cnt_w_re = 0
            for name, para in model_train.named_parameters():
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
                    modi_w = mask[name] > 0
                    num_modi_re = modi_w.sum()
                    cnt_w_re += num_modi_re
            if not fp16:
                outputs1, outputs2 = model_train(images, "train")

                _triplet_loss   = loss(outputs1, Batch_size)
                _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
                _loss           = _triplet_loss + _CE_loss
                _loss = -_loss + 2 ** (kld * 1.5e2)
                _loss.backward()
                for name, para in model_train.named_parameters():
                    if name in mask:
                        para.grad *= mask[name].long()
                    else:
                        para.grad *= 0

                optimizer.step()
                with torch.no_grad():
                    for name, para in model_train.named_parameters():
                        if name in mask:
                            para.data = (1 - mask[name]) * para + (mask[name] * para).clamp_(ori_min[name] * 0.85,
                                                                                             ori_max[name] * 0.75)
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs1, outputs2 = model_train(images, "train")

                    _triplet_loss   = loss(outputs1, Batch_size)
                    _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
                    _loss           = _triplet_loss + _CE_loss
                scaler.scale(_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                accuracy         = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            total_triple_loss   += _triplet_loss.item()
            total_CE_loss       += _CE_loss.item()
            total_accuracy      += accuracy.item()

            if local_rank == 0:
                pbar.set_postfix(**{'total_triple_loss' : total_triple_loss / (iteration + 1),
                                    'total_CE_loss'     : total_CE_loss / (iteration + 1),
                                    'accuracy'          : total_accuracy / (iteration + 1),
                                    'lr'                : get_lr(optimizer)})
                pbar.update(1)

        if local_rank == 0:
            pbar.close()
            print('Finish Train')
            print('Start Validation')
            pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

        min_w = []
        max_w = []
        cnt_w_re = 0
        layer_modi_re = []
        cos_sim = 0
        cnt = 0
        kld = 0
        total_cnt = 0
        for name, para in model_train.named_parameters():
            if len(para.shape) > 1 and name in mask:
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
                kld += F.kl_div(torch.log_softmax(flattened_para, dim=0), torch.softmax(flattened_ori, dim=0),
                                reduction='sum')
                cnt += 1

        min_w_v = min(min_w)
        max_w_v = max(max_w)
        kld = 2 ** (kld * 2e2)
        print('============cos: %.5f' % (cos_sim / cnt))
        print('============kld: %f' % kld)
        print(
            '==========================[epoch:%d] Loss: %.03f | ori num:%d \n| cur num:%d'
            % (epoch + 1, _loss.item(),  cnt_w, cnt_w_re))

        print('==========================min: %.5f | max: %.5f '
              % (min_w_v, max_w_v))

        # for name, module in model_train.named_modules():
        #     if isinstance(module, torch.nn.Conv2d):
        #         if 'branch0' in name or '3.2.branch1.2' in name or '3.3.branch1.2' in name:
        #             continue
        #         draw(module.weight.data, name, mask)

        if lfw_eval_flag:
            print("LFW Validation。")
            labels, distances = [], []
            for _, (data_a, data_p, label) in enumerate(test_loader):
                with torch.no_grad():
                    data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                    if cuda:
                        data_a, data_p = data_a.cuda(local_rank), data_p.cuda(local_rank)
                    out_a, out_p = model_train(data_a), model_train(data_p)
                    dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                distances.append(dists.data.cpu().numpy())
                labels.append(label.data.cpu().numpy())

            labels      = np.array([sublabel for label in labels for sublabel in label])
            distances   = np.array([subdist for dist in distances for subdist in dist])
            _, _, accuracy, _, _, _, _ = evaluate(distances,labels)

        if local_rank == 0:
            pbar.close()
            print('Finish Validation')

            if lfw_eval_flag:
                print('LFW_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

            loss_history.append_loss(epoch, np.mean(accuracy) if lfw_eval_flag else total_accuracy / epoch_step, \
                (total_triple_loss + total_CE_loss) / epoch_step, (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)
            print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
            print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))
            if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch or np.mean(accuracy) < 0.57:
                torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1),
                                                                        (total_triple_loss + total_CE_loss) / epoch_step,
                                                                        (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)))
        my_op = 1
        if my_op == 1:
            if np.mean(accuracy) < 0.57:
                torch.save(model_train.state_dict(), f'./distshield_facenet.pth')
                step = 0.1
                with torch.no_grad():
                    for name, module in model_train.named_modules():
                        name = name + '.weight'
                        if isinstance(module, torch.nn.Conv2d):
                            if 'branch0' in name or '3.2.branch1.2' in name or '3.3.branch1.2' in name:
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
                            weights[zero_mask.view(mask[name].size()) > 0] = ori_dict[name][
                                zero_mask.view(mask[name].size()) > 0]
                re_op = 1
                if re_op == 1:
                    optimizer = {
                        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                                           weight_decay=weight_decay),
                        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                                         weight_decay=weight_decay)
                    }[optimizer_type]