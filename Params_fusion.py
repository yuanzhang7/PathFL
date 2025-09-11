import numpy as np
from scipy import spatial
import torch
import copy
import os
import random
from tqdm import tqdm

from Network import UNet
from Train_Utils import set_random_seed
import time

set_random_seed(3407)

def safe_load_model(path, max_retry=100):
    """
    安全加载模型，如果遇到 RuntimeError 或 ValueError，则等待并重试。
    """
    for attempt in range(max_retry):
        try:
            return torch.load(path, map_location='cpu')
        except (RuntimeError, ValueError, EOFError) as e:
            print(f"Error loading model: {str(e)}")
            print(f"Retrying ({attempt + 1}/{max_retry})...")
            time.sleep(1)
    raise RuntimeError(f"Failed to load model after {max_retry} attempts. Path: {path}")


def output_middle_layer(net, x):
    input_data = {}
    output_data = {}
    outputs = []

    def get_activation(name):
        def hook(model, input, output):
            input_data[name] = input[0].detach()
            output_data[name] = output.detach()
        return hook

    target_layers = ['conv1', 'conv3', 'conv5', 'conv7', 'conv9', 'conv11', 'conv13', 'out_conv']
    for layer_name in target_layers:
        getattr(net, layer_name).register_forward_hook(get_activation(layer_name))

    y = net(torch.from_numpy(x).cuda())
    for layer_name in target_layers:
        outputs.append(output_data[layer_name].flatten())
    return outputs


def update_and_save_input(mean, std, save_path, num, roi_shape):
    new_sample = np.random.normal(loc=mean.reshape(3, 1, 1), scale=std.reshape(3, 1, 1),
                                  size=(num, 3, roi_shape, roi_shape)).astype(np.float32)
    if os.path.exists(save_path):
        former_sample = np.load(save_path)
        updated_sample = new_sample / 4 + 3 * former_sample / 4
    else:
        updated_sample = new_sample
    np.save(save_path, updated_sample)
    return updated_sample


def For_one_centre(C_list, args, net):
    mean0, std0 = np.load(args.Meanstd_dir)
    mean1, std1 = np.load(args.Meanstd_dir1)
    mean2, std2 = np.load(args.Meanstd_dir2)

    num = 4
    x_in_0 = update_and_save_input(mean0, std0, args.save_path_array + 'x_in_0.npy', num, args.ROI_shape)
    x_in_1 = update_and_save_input(mean1, std1, args.save_path_array + 'x_in_1.npy', num, args.ROI_shape)
    x_in_2 = update_and_save_input(mean2, std2, args.save_path_array + 'x_in_2.npy', num, args.ROI_shape)

    pbar = tqdm(total=24, desc='Cosine Similarity Calculation')

    cos_sim_all = []
    for index in range(3):
        cos_sim_single = []
        net.load_state_dict(C_list[index])
        net = net.cuda()

        outputs_list0 = output_middle_layer(net, x_in_0)
        outputs_list1 = output_middle_layer(net, x_in_1)
        outputs_list2 = output_middle_layer(net, x_in_2)

        for ind in range(8):
            x_out = [
                outputs_list0[ind].data.cpu().numpy().flatten(),
                outputs_list1[ind].data.cpu().numpy().flatten(),
                outputs_list2[ind].data.cpu().numpy().flatten(),
            ]
            cos_sim = [
                1 - spatial.distance.cosine(x_out[index], x_out[0]),
                1 - spatial.distance.cosine(x_out[index], x_out[1]),
                1 - spatial.distance.cosine(x_out[index], x_out[2]),
            ]
            cos_sim.pop(index)
            cos_sim_single.append(cos_sim)
            pbar.update(1)
        cos_sim_all.append(cos_sim_single)
    pbar.close()
    return cos_sim_all


def fusion(C_list, w, index, k):
    w1, w2 = w
    w1_new = w1 / (w1 + w2)  # 恢复旧版本逻辑，不使用denom检查
    w2_new = w2 / (w1 + w2)

    dynamic_w = 0.4
    C_list_copy = [c for i, c in enumerate(C_list) if i != index]
    out = (1 - dynamic_w) * C_list[index][k].data
    out += dynamic_w * w1_new * C_list_copy[0][k].data
    out += dynamic_w * w2_new * C_list_copy[1][k].data
    return out


def Fuse(dir, input_param_list, output_param_list, args, net):
    with torch.no_grad():
        # C_list = [
        #     torch.load(dir + input_param_list[1], map_location='cpu'),
        #     torch.load(dir + input_param_list[2], map_location='cpu'),
        #     torch.load(dir + input_param_list[3], map_location='cpu'),
        # ]
        C_list = [
            safe_load_model(os.path.join(dir, input_param_list[1])),
            safe_load_model(os.path.join(dir, input_param_list[2])),
            safe_load_model(os.path.join(dir, input_param_list[3])),
        ]
        C_avg = copy.deepcopy(C_list[0])
        cos_sim_all = For_one_centre(C_list, args, net)
        C_list_new = [copy.deepcopy(c) for c in C_list]

        total_layers = len(C_list[0].keys())
        with tqdm(total=total_layers * 3, desc='Parameter Fusion') as pbar:
            for count, (k, _) in enumerate(C_list[0].items()):
                group_idx = min(count // 8, 7)
                for index in range(3):
                    w = cos_sim_all[index][group_idx]
                    out = fusion(C_list, w, index, k)
                    C_list_new[index][k].data = out
                    pbar.update(1)
                for i in range(3):
                    C_list[i][k].data = C_list_new[i][k].data

        for k in C_avg.keys():
            C_avg[k].data = (C_list[0][k].data.float() + C_list[1][k].data.float() + C_list[2][k].data.float()) / 3
            for i in range(3):
                C_list[i][k].data = C_avg[k].data

        torch.save(C_avg, dir + output_param_list[0])
        torch.save(C_list[0], dir + output_param_list[1])
        torch.save(C_list[1], dir + output_param_list[2])
        torch.save(C_list[2], dir + output_param_list[3])

