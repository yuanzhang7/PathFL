import os
import torch
import numpy as np
from Network import UNet
from Params_fusion import Fuse
from Train_Utils import train_epoch, Train_net, predict, Dice, Create_files, Get_logger, set_random_seed
import argparse

def Predict_Network(net, args, center_id):
    print(f"\n--- Prediction for Center {center_id} ---")
    
    # GPU检测和设置
    if torch.cuda.is_available():
        net = net.cuda()
        print(f"Model moved to GPU: {next(net.parameters()).device}")
        if torch.cuda.is_available():
            print(f"GPU memory before prediction: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("WARNING: CUDA not available, prediction on CPU!")
        print(f"Model device: {next(net.parameters()).device}")
    
    checkpoint_dir_c = getattr(args, f'checkpoint_dir_c{center_id}')
    test_image_txt = getattr(args, f'c{center_id}_test_image_txt')
    test_label_txt = getattr(args, f'c{center_id}_test_label_txt')
    
    # 尝试加载最佳模型
    best_model_path = os.path.join(checkpoint_dir_c, args.model_name_max)
    if os.path.exists(best_model_path):
        try:
            net.load_state_dict(torch.load(best_model_path))
            print(f"Successfully loaded best model: {best_model_path}")
            print(f"Best model file size: {os.path.getsize(best_model_path) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Falling back to regular model...")
            net.load_state_dict(torch.load(os.path.join(checkpoint_dir_c, args.model_name)))
            print(f"Loaded regular model: {os.path.join(checkpoint_dir_c, args.model_name)}")
    else:
        print(f"Best model not found: {best_model_path}")
        print("Loading regular model instead...")
        net.load_state_dict(torch.load(os.path.join(checkpoint_dir_c, args.model_name)))
        print(f"Loaded regular model: {os.path.join(checkpoint_dir_c, args.model_name)}")
    
    # 检查加载后的模型设备
    print(f"Model device after loading: {next(net.parameters()).device}")
    
    predict(net, test_image_txt, args.save_path_max, args)
    
    # 使用与旧版本一致的Dice调用方式，传递base_dir参数
    if center_id == 0:
        dice = Dice(args.c0_test_label_txt, args.save_path_max, args.base_dir)
    elif center_id == 1:
        dice = Dice(args.c1_test_label_txt, args.save_path_max, args.base_dir)
    elif center_id == 2:
        dice = Dice(args.c2_test_label_txt, args.save_path_max, args.base_dir)
    
    dice_mean_final = np.mean(dice)
    print(f"Final test dice: {dice_mean_final:.4f}")
    
    if torch.cuda.is_available():
        print(f"GPU memory after prediction: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # 验证最佳模型是否真的是最佳
    print(f"Final prediction completed with dice: {dice_mean_final:.4f}")
    if hasattr(args, 'dice_m'):
        print(f"Training best dice was: {args.dice_m:.4f}")
        if dice_mean_final < args.dice_m:
            print("WARNING: Final prediction dice is lower than training best dice!")
            print("This might indicate the best model was not properly saved or loaded.")
        else:
            print("✓ Final prediction dice matches or exceeds training best dice.")

def FederatedProcess(net, args, center_id):
    import time
    from datetime import datetime
    log_name = f"{datetime.today().date()}_{datetime.today().time().hour}-{datetime.today().time().minute}-{datetime.today().time().second}_C{center_id}_{args.log_name}"
    logger = Get_logger(os.path.join(args.log_dir, log_name))
    logger.info('start training!')
    
    print(f"\n--- Federated Training for Center {center_id} ---")
    
    # 检查模型初始设备
    if hasattr(net, 'parameters'):
        initial_device = next(net.parameters()).device
        print(f"Initial model device: {initial_device}")
    
    # 初始化最佳dice记录
    args.dice_m = 0
    print(f"Initialized best dice record: {args.dice_m}")

    checkpoint_dir_c = getattr(args, f'checkpoint_dir_c{center_id}')
    model_name = args.model_name
    # 只有明确设置if_retrain=False时才加载已有模型
    if not args.if_retrain:
        model_path = os.path.join(checkpoint_dir_c, model_name)
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path))
            print(f"Loaded checkpoint from {model_path}")
    else:
        Train_net(net, args, model_name, logger, step=0, per_epoch=5, center_id=center_id)  # Warm up

    for step in range(args.start_train_step, args.n_epochs // args.n_epoch_per_step):
        step_model_name = args.model_name + '_step' + str(step)
        torch.save(net.state_dict(), os.path.join(checkpoint_dir_c, step_model_name))
        update_model_name = args.model_name + '_update' + str(step)

        if center_id == 0:
            # 只有中心0负责融合
            c0_model_dir = os.path.join(args.checkpoint_dir_c0, step_model_name)
            c1_model_dir = os.path.join(args.checkpoint_dir_c1, step_model_name)
            c2_model_dir = os.path.join(args.checkpoint_dir_c2, step_model_name)
            input_param_list = ['Server/' + step_model_name] + [f'C{i}/{step_model_name}' for i in range(3)]
            output_param_list = ['Server/' + update_model_name] + [f'C{i}/{update_model_name}' for i in range(3)]
            count = 0
            while not (os.path.exists(c0_model_dir) and os.path.exists(c1_model_dir) and os.path.exists(c2_model_dir)):
                print("Now waiting for the parameters ...")
                count += 1
                time.sleep(5)
            print(f"Now waiting for the {count * 5} seconds ...")
            Fuse(args.checkpoint_dir, input_param_list, output_param_list, args, net)
        else:
            model_dir = os.path.join(checkpoint_dir_c, update_model_name)
            count = 0
            while not os.path.exists(model_dir):
                print("Now waiting for the parameters ...")
                count += 1
                time.sleep(5)
            print(f"Now waiting for the {count * 5} seconds ...")

        Train_net(net, args, update_model_name, logger, step, args.n_epoch_per_step, center_id=center_id)

        # 每轮都测试并保存最佳模型（恢复旧版本逻辑）
        if step >= args.start_verify_step:
            net.load_state_dict(torch.load(os.path.join(checkpoint_dir_c, update_model_name)))
            # 根据center_id获取对应的测试图像文本文件路径
            if center_id == 0:
                test_image_txt = args.c0_test_image_txt
                test_label_txt = args.c0_test_label_txt
            elif center_id == 1:
                test_image_txt = args.c1_test_image_txt
                test_label_txt = args.c1_test_label_txt
            elif center_id == 2:
                test_image_txt = args.c2_test_image_txt
                test_label_txt = args.c2_test_label_txt
            else:
                print(f"ERROR: Invalid center_id: {center_id}")
                continue
            
            # 先进行预测
            predict(net, test_image_txt, args.save_path, args)
            # 然后计算dice分数
            dice = Dice(test_label_txt, args.save_path, args.base_dir)
            dice_mean = np.mean(dice)
            
            # 保存最佳模型 - 修复逻辑：只有当当前dice更好时才保存最佳模型
            if not hasattr(args, 'dice_m'):
                args.dice_m = 0
            if dice_mean > args.dice_m:
                prev = args.dice_m
                args.dice_m = dice_mean
                # 保存当前最佳模型状态
                torch.save(net.state_dict(), os.path.join(checkpoint_dir_c, args.model_name_max))
                print(f"New best model saved! Dice improved from {prev:.4f} to {dice_mean:.4f}")
            else:
                print(f"Current dice {dice_mean:.4f} not better than best {args.dice_m:.4f}, keeping previous best model")
            
            logger.info(f'dice_mean={dice_mean:.4f} max_dice={args.dice_m:.4f}')
    logger.info('finish training!')

def Process(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    set_random_seed(3407)
    
    # 详细的GPU检测
    print("=" * 50)
    print("GPU DETECTION AND SETUP")
    print("=" * 50)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_properties = torch.cuda.get_device_properties(current_device)
        
        print(f"CUDA is available!")
        print(f"Number of GPUs: {device_count}")
        print(f"Current GPU device: {current_device}")
        print(f"GPU name: {device_name}")
        print(f"GPU memory: {device_properties.total_memory / 1024**3:.2f} GB")
        print(f"GPU compute capability: {device_properties.major}.{device_properties.minor}")
        
        # 检查GPU内存使用情况
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2
        print(f"GPU memory allocated: {memory_allocated:.2f} MB")
        print(f"GPU memory reserved: {memory_reserved:.2f} MB")
    else:
        print("WARNING: CUDA is not available! Training will be on CPU.")
        print("This may significantly slow down training.")
    
    print("=" * 50)
    
    net = UNet(args)
    
    # 检查网络初始化后的设备
    if hasattr(net, 'parameters'):
        net_device = next(net.parameters()).device
        print(f"Network initialized on device: {net_device}")
    
    Create_files(args)
    
    # 检查关键文件是否存在
    print("=" * 50)
    print("CHECKING CRITICAL FILES")
    print("=" * 50)
    
    # 检查训练数据文件
    train_image_txt = getattr(args, f'c{args.center_id}_train_image_txt')
    train_label_txt = getattr(args, f'c{args.center_id}_train_label_txt')
    test_image_txt = getattr(args, f'c{args.center_id}_test_image_txt')
    test_label_txt = getattr(args, f'c{args.center_id}_test_label_txt')
    
    for file_path in [train_image_txt, train_label_txt, test_image_txt, test_label_txt]:
        if not os.path.exists(file_path):
            print(f"WARNING: File not found: {file_path}")
        else:
            print(f"✓ Found: {file_path}")
    
    # 检查meanstd文件
    if not os.path.exists(args.Meanstd_dir):
        print(f"WARNING: Meanstd file not found: {args.Meanstd_dir}")
    else:
        print(f"✓ Found: {args.Meanstd_dir}")
    
    print("=" * 50)
    
    if not args.if_onlytest:
        FederatedProcess(net, args, center_id=args.center_id)
        Predict_Network(net, args, center_id=args.center_id)
    else:
        Predict_Network(net, args, center_id=args.center_id)

def get_center_config(center_id, base_dir, model_name, txt_dir):
    if not base_dir.endswith('/'):
        base_dir += '/'
    if not txt_dir.endswith('/'):
        txt_dir += '/'
    return {
        'train_image_txt': f'{base_dir}{txt_dir}c{center_id}_train_image.txt',
        'train_label_txt': f'{base_dir}{txt_dir}c{center_id}_train_label.txt',
        'test_image_txt': f'{base_dir}{txt_dir}c{center_id}_test_image.txt',
        'test_label_txt': f'{base_dir}{txt_dir}c{center_id}_test_label.txt',
        'meanstd': f'{base_dir}{txt_dir}c{center_id}_train_image_meanstd.npy',
        'save_path': f'{base_dir}Results/Cell/{model_name}/C{center_id}/',
        'save_path_max': f'{base_dir}Results/Cell/{model_name}/C{center_id}_max/',
        'save_path_array': f'{base_dir}Array/Cell/{model_name}/',
        'checkpoint_dir': f'{base_dir}Weights/Cell/{model_name}/',
        'checkpoint_dir_c': f'{base_dir}Weights/Cell/{model_name}/C{center_id}/',
        'model_name': model_name,
        'model_name_max': f'{model_name}_max',
        'log_name': f'{model_name}_C{center_id}.log',
        'log_dir': f'{base_dir}Log/Cell/{model_name}/',
        'style_folder': f'{base_dir}Style/{model_name}/',
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--center_id', type=int, required=True, help='中心编号 0/1/2')
    parser.add_argument('--base_dir', default='./', help='Project root directory')
    parser.add_argument('--txt_dir', default='Txt/ClusterSeg_Txt', help='txt文件目录')
    parser.add_argument('--GPU_id', default='0', help='GPU ID')
    parser.add_argument('--model_name', default='PathFL_a7', help='Model name prefix')
    parser.add_argument('--n_channels', default=3, type=int, help='input channels')
    parser.add_argument('--n_classes', default=1, type=int, help='output channels')
    parser.add_argument('--n_basic_layer', default=32, type=int, help='basic layer numbers')
    parser.add_argument('--ROI_shape', default=224, type=int, help='roi size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--n_epochs', default=500, type=int, help='Epoch Num')
    parser.add_argument('--if_onlytest', default=False, type=bool, help='If Only Test')
    parser.add_argument('--if_retrain', default=True, type=bool, help='If Retrain') 
    parser.add_argument('--start_train_step', default=0, type=int, help='Start training step')
    parser.add_argument('--start_verify_step', default=10, type=int, help='Start verifying step')
    parser.add_argument('--n_epoch_per_step', default=5, type=int, help='N epoch per step')
    # 如有其它参数需求可继续补充
    args, unknown = parser.parse_known_args()

    center_cfg = get_center_config(args.center_id, args.base_dir, args.model_name, args.txt_dir)

    # 先动态赋值
    args.c_train_image_txt = center_cfg['train_image_txt']
    args.c_train_label_txt = center_cfg['train_label_txt']
    args.test_image_txt = center_cfg['test_image_txt']
    args.test_label_txt = center_cfg['test_label_txt']
    args.Meanstd_dir = center_cfg['meanstd']
    args.save_path = center_cfg['save_path']
    args.save_path_max = center_cfg['save_path_max']
    args.save_path_array = center_cfg['save_path_array']
    args.checkpoint_dir = center_cfg['checkpoint_dir']
    args.checkpoint_dir_c = center_cfg['checkpoint_dir_c']
    args.model_name = center_cfg['model_name']
    args.model_name_max = center_cfg['model_name_max']
    args.log_name = center_cfg['log_name']
    args.log_dir = center_cfg['log_dir']
    args.style_folder = center_cfg['style_folder']

    # 为所有中心都赋值 checkpoint_dir_c0/1/2
    args.checkpoint_dir_c0 = get_center_config(0, args.base_dir, args.model_name, args.txt_dir)['checkpoint_dir_c']
    args.checkpoint_dir_c1 = get_center_config(1, args.base_dir, args.model_name, args.txt_dir)['checkpoint_dir_c']
    args.checkpoint_dir_c2 = get_center_config(2, args.base_dir, args.model_name, args.txt_dir)['checkpoint_dir_c']

    # 为所有中心都赋值 Meanstd_dir0/1/2
    args.Meanstd_dir0 = get_center_config(0, args.base_dir, args.model_name, args.txt_dir)['meanstd']
    args.Meanstd_dir1 = get_center_config(1, args.base_dir, args.model_name, args.txt_dir)['meanstd']
    args.Meanstd_dir2 = get_center_config(2, args.base_dir, args.model_name, args.txt_dir)['meanstd']

    # 兼容各中心训练脚本的参数名
    if args.center_id == 0:
        args.c0_train_image_txt = args.c_train_image_txt
        args.c0_train_label_txt = args.c_train_label_txt
        args.c0_test_image_txt = args.test_image_txt
        args.c0_test_label_txt = args.test_label_txt
    elif args.center_id == 1:
        args.c1_train_image_txt = args.c_train_image_txt
        args.c1_train_label_txt = args.c_train_label_txt
        args.c1_test_image_txt = args.test_image_txt
        args.c1_test_label_txt = args.test_label_txt
    elif args.center_id == 2:
        args.c2_train_image_txt = args.c_train_image_txt
        args.c2_train_label_txt = args.c_train_label_txt
        args.c2_test_image_txt = args.test_image_txt
        args.c2_test_label_txt = args.test_label_txt

    Process(args)
