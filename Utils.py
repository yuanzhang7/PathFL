import csv
from sklearn.cluster import KMeans
import os
import torch
import time
import numpy as np
import random


def read_epoch_mean_std(style_folder,current_client,epoch):
    file_path_my = style_folder + 'C' + str(current_client) + '_epoch' + str(epoch - 1) + '.csv'
    try:
        with open(file_path_my, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            row = next(reader)
            if len(row) >= 2:
                mean_my, std_my = row[0], row[1]
            else:
                # 如果格式不正确，使用默认值
                mean_my, std_my = "0.0", "1.0"
    except (FileNotFoundError, StopIteration, IndexError):
        # 如果文件不存在或格式错误，使用默认值
        mean_my, std_my = "0.0", "1.0"
    ## 读取其他参数的均值和方差
    mean_list = []
    std_list = []
    for index in range(3):
        file_path1 = style_folder + 'C' + str(index) + '_epoch' + str(epoch - 1) + '.csv'
        try:
            if os.path.exists(file_path1):
                with open(file_path1, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)  # 跳过标题行
                    row = next(reader)
                    if len(row) >= 2:
                        mean_values, std_values = row[0], row[1]
                    else:
                        # 如果格式不正确，使用默认值
                        mean_values, std_values = "0.0", "1.0"
                    mean_list.append(float(mean_values))
                    std_list.append(float(std_values))
            else:
                # 如果文件不存在，使用默认值
                mean_list.append(0.0)
                std_list.append(1.0)
        except (FileNotFoundError, StopIteration, IndexError):
            # 如果读取失败，使用默认值
            mean_list.append(0.0)
            std_list.append(1.0)

    mean_my = torch.tensor(float(mean_my))
    std_my = torch.tensor(float(std_my))

    mean_list = torch.tensor([float(value) for value in mean_list])
    std_list = torch.tensor([float(value) for value in std_list])
    return mean_my, std_my, mean_list, std_list


def GetMeanStd(input_tensor):
    mean = torch.mean(input_tensor)
    variance = torch.var(input_tensor)
    std= torch.sqrt(variance)
    # std=variance.sqrt()
    return mean,std

def GetKmeans(n_clusters,input_data):
    # n_clusters = 1
    # 使用K均值算法进行聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(input_data)
    # 获取聚类中心的均值和方差
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def read_deep_mean_std(style_folder,epoch):
    mean_list = []
    std_list = []
    for index in range(3):
        file_path1 = style_folder + 'C' + str(index) + '_epoch' + str(epoch - 1) + '_deep.csv'
        try:
            if os.path.exists(file_path1):
                with open(file_path1, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)  # 跳过标题行
                    data_values = []
                    for row in reader:
                        if len(row) >= 2:
                            data_values.append([float(row[0]), float(row[1])])
                        else:
                            # 如果格式不正确，使用默认值
                            data_values.append([0.0, 1.0])
                    
                    if data_values:
                        # 将data_values转换为NumPy数组以便于计算
                        data_array = np.array(data_values)
                        # 计算所有mean的均值和std的均值
                        mean_avg = np.mean(data_array[:, 0])
                        std_avg = np.mean(data_array[:, 1])
                    else:
                        # 如果没有有效数据，使用默认值
                        mean_avg = 0.0
                        std_avg = 1.0

                    mean_list.append(mean_avg)
                    std_list.append(std_avg)
            else:
                # 如果文件不存在，使用默认值
                mean_list.append(0.0)
                std_list.append(1.0)
        except (FileNotFoundError, StopIteration, IndexError, ValueError):
            # 如果读取失败，使用默认值
            mean_list.append(0.0)
            std_list.append(1.0)
    mean_of_mean_list = sum(mean_list) / len(mean_list)
    mean_of_std_list = sum(std_list) / len(std_list)
    mean_client = torch.tensor(float(mean_of_mean_list))
    std_client = torch.tensor(float(mean_of_std_list))
    return mean_client, std_client

def HalfMask(I1,I2,imagesize=512, batchsize=4, channels=3):
    """
    生成一个随机的格子mask张量。

    参数:
    size (int): 输出张量的大小 (默认为512)
    grid_size (int): 格子的大小 (默认为32)
    I1&I2 (4,3,512,512)
    返回:
    torch.Tensor: 大小为 (4,3,512,512)
    """

    # is_horizontal = random.choice([True, False])
    mask = torch.zeros((imagesize, imagesize), device=I1.device)
    # 随机选择mask类型
    mask_type = random.randint(0, 3)

    if mask_type == 0:
        mask[:imagesize // 2, :] = 1
    elif mask_type == 1:
        mask[imagesize // 2:, :] = 1
    elif mask_type == 2:
        mask[:, :imagesize // 2] = 1
    else:  # mask_type == 3
        mask[:, imagesize // 2:] = 1

    # 扩展到指定的通道数
    if I1.shape[0]==batchsize:
        mask = mask.unsqueeze(0).repeat(batchsize,channels, 1, 1)#[batchsize,channels,512,512]
    else:
        mask = mask.unsqueeze(0).repeat(I1.shape[0], channels, 1, 1)  # [batchsize,channels,512,512]
    mask=mask.cuda()
    I_mix = I1 * mask + I2 * (1 - mask)
    I_mix = I_mix.cuda()
    return I_mix