import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pickle
from torchvision import transforms
from PyTorch_CIFAR10_master.cifar10_models.resnet import resnet18
from PyTorch_CIFAR10_master.cifar10_models.vgg import vgg11_bn
from PyTorch_CIFAR10_master.cifar10_models.mobilenetv2 import mobilenet_v2

device = torch.device("cpu")

def get_layer_params(model, layer_name):
    layers = dict(model.named_modules())
    if layer_name in layers:
        layer = layers[layer_name]
        return layer.weight.data
    else:
        raise ValueError(f"Layer {layer_name} not found in the model.")

def compare_weights(net, pretrained_weights, layer_name):
    model_weights = get_layer_params(net, layer_name)
    diff = model_weights - pretrained_weights
    return diff


def generate_sparse_matrix(diff):
    batch_size, channels, height, width = diff.size()
    diff_flat = diff.view(batch_size, channels * height * width)
    indices = diff_flat.nonzero().t()
    values = diff_flat[indices[0], indices[1]]
    sparse_tensor = torch.sparse_coo_tensor(indices, values, diff_flat.size()).coalesce()
    return sparse_tensor


def sparse_matmul(sparse_matrix, dense_matrix):
    batch_size, channels, height, width = dense_matrix.size()
    dense_matrix_flat = dense_matrix.view(batch_size * channels, height * width)
    result_flat = torch.sparse.mm(sparse_matrix, dense_matrix_flat)
    result = result_flat.view(batch_size, channels, height, width)
    return result


def measure_time(sparse_matrix, dense_matrix, num_runs=256):
    start_time = time.time()
    for _ in range(num_runs):
        _ = torch.sparse.mm(sparse_matrix, dense_matrix)
    end_time = time.time()
    total_time = (end_time - start_time)
    return total_time



def count_nonzero_elements(sparse_matrix):
    return sparse_matrix._nnz()


def cluster_vectors(vectors, cluster_size=4):
    index_pairs = np.array([np.array([i]) for i in range(len(vectors))])
    iter = 1
    while(iter<cluster_size):
        iter*=2
        cos_sim_matrix = cosine_similarity(vectors)
        sum_cos_sim_dis = np.mean(cos_sim_matrix, axis=0)
        sorted_indices = np.argsort(sum_cos_sim_dis)[::-1]
        np.fill_diagonal(cos_sim_matrix, np.inf)
        pairs = []
        index_pair = []
        repeat_index = []
        for i in sorted_indices:
            if i in repeat_index:
                continue
            j = np.argmin(cos_sim_matrix[i])
            repeat_index.append(j)
            cos_sim_matrix[i, :] = np.inf
            cos_sim_matrix[:, i] = np.inf
            cos_sim_matrix[j, :] = np.inf
            cos_sim_matrix[:, j] = np.inf
            index_pair.append(np.concatenate((index_pairs[i], index_pairs[j])))
            pairs.append(np.mean([vectors[i], vectors[j]],axis=0))
        vectors = pairs
        index_pairs = index_pair
        # print(index_pair)
    return index_pairs


def modify_conv_layers(original_model, cluster_size=4):
    modified_model = copy.deepcopy(original_model)
    device = torch.device('cuda')
    restore_params = {}

    for layer_name, layer in modified_model.named_modules():
        if isinstance(layer, nn.Conv2d):

            with torch.no_grad():
                out_channels, in_channels, kernel_height, kernel_width = layer.weight.shape
                weights = torch.zeros((out_channels, in_channels * kernel_height * kernel_width))
                for i in range(out_channels):
                    mod_weight = layer.weight[i, :, :, :].view(in_channels, -1).flatten()
                    weights[i] = mod_weight
                # print("{:.10e}".format(layer.weight.data[0,1,0,0].item()))

                cluster_index = cluster_vectors(weights.detach().numpy(), cluster_size=cluster_size)
                random_coeff_list = [[] for _ in range(out_channels)]
                inv_A_list = []
                for idlist in cluster_index:
                    new_kernels = []
                    for i in idlist:
                        random_coeffs = np.random.randint(1, 100, size=cluster_size)
                        random_coeff_list[i] = random_coeffs

                        new_kernel = sum(
                            coeff * layer.weight[idlist[j], :, :, :] for j, coeff in enumerate(random_coeffs))
                        new_kernels.append(new_kernel)

                    for index, idx in enumerate(idlist):
                        layer.weight.data[idx, :, :, :] = new_kernels[index]
                for idlist in cluster_index:
                    A = []
                    for i in idlist:
                        A.append(np.array(random_coeff_list[i]))
                    A = np.array(A, dtype=np.float64)
                    inv_A = np.linalg.inv(A)
                    inv_A_list.append(torch.tensor(inv_A, dtype=torch.float32))

                perm = torch.randperm(in_channels)
                layer.weight.data = layer.weight.data[:, perm, :, :]

                restore_params[layer_name] = {
                        'shuffle_indices': perm,
                        'cluster_index': cluster_index,
                        'inv_A': inv_A_list
                    }


    return modified_model, restore_params


def worker1_process_layer(x, layer_fullname, model):
    layer = dict(model.named_modules())[layer_fullname]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        layer.to(device)
        x = x.to(device)
    else:
        device = torch.device("cpu")
    with torch.no_grad():
        x = layer(x)
    x = x.to(torch.device("cpu"))
    return x


if __name__ == '__main__':
    # 加载预训练模型和本地模型
    net_pre = resnet18(pretrained=True).to(device)
    net = resnet18().to(device)
    net_nns = resnet18().to(device)
    net_mag = resnet18().to(device)

    pth_file_path = 'my_c10_mobile_op.pth'
    net.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
    pth_file_path = 'nns_c10_mobile.pth'
    net_nns.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
    pth_file_path = 'mag_mob_c10.pth'
    net_mag.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))

    time_used = {
        'my ': 0,
        'nns': 0,
        'mag': 0,
    }
    nonzero = {
        'my ': 0,
        'nns': 0,
        'mag': 0,
    }
    for model, name in zip([net_mag, net_nns, net], ['mag', 'nns', 'my ']):
        for layer_name, module in net_pre.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                diff = compare_weights(model, get_layer_params(net_pre, layer_name), layer_name)
                sparse_matrix = generate_sparse_matrix(diff)
                with open(f'layers/{layer_name}.pkl', 'rb') as f:
                    dense_matrix = torch.from_numpy(pickle.load(f))
                time_taken = 0
                nonzero_elements = count_nonzero_elements(sparse_matrix)
                time_used[name] += time_taken
                nonzero[name] += nonzero_elements
        print(f'{name}:{time_used[name]:.7f}s, {nonzero[name]}weights')
        print('---------------------')

    # groupcover
    modified_model, restore_params = modify_conv_layers(net_pre, cluster_size=4)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if inputs.shape[0] < 128:
            print(inputs.shape[0])
            break
        x = inputs
        time_cost = 0
        for fullname, layer in net_pre.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if fullname in dense_matrix_dict:
                    x = worker1_process_layer(x, fullname, net_pre)
                    start_time = time.time()
                    original_shape = x.shape
                    C = original_shape[1]
                    # N,C,H,W = original_shape
                    # x -= restore_params[fullname]['Wr']
                    # x_reshaped = x.view(N, C, H*W)
                    groups = restore_params[fullname]['cluster_index']
                    groups = torch.tensor(groups, dtype=torch.long)
                    cof_mat = restore_params[fullname]['inv_A']
                    x = x.view(C, -1)
                    cof_tensor = torch.stack([torch.tensor(cof_mat[i][j]) for j in range(4) for i in range(C // 4)])
                    group_indices = torch.tensor(groups)
                    for i in range(C // 4):
                        group_idx = group_indices[i]
                        cof = cof_tensor[i]
                        x[[i, i + 1, i + 2, i + 3], :] = torch.sum(x[group_idx, :] * cof[:, None], dim=0)
                    x = x.view(*original_shape)
                    end_time = time.time()
                    time_cost += (end_time - start_time)
                    # print((end_time - start_time))
        break
    print(time_cost)





