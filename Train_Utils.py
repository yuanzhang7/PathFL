import os 
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Loss import cross_loss
from MyDataloader import Dataloader
from Utils import read_epoch_mean_std, GetMeanStd, read_deep_mean_std, HalfMask
from PIL import Image
import logging

def set_random_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed()

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def step_optimization(model, optimizer, criterion, image, label, loss_meter):
    model_device = next(model.parameters()).device
    if model_device != image.device or model_device != label.device:
        print(f"WARNING: Device mismatch! model:{model_device}, image:{image.device}, label:{label.device}")
    optimizer.zero_grad()
    output = model(image)

    if output.shape != label.shape:
        print(f"WARNING: Output shape {output.shape} != Label shape {label.shape}")

    lmin, lmax = label.min().item(), label.max().item()
    if lmin < 0 or lmax > 1:
        print(f"WARNING: Label values out of [0,1], min={lmin}, max={lmax}")

    loss = criterion(label, output)
    loss_meter.update(loss.item(), label.size(0))
    loss.backward()
    optimizer.step()

def train_epoch(model, loader, optimizer, criterion, epoch, style_folder, center_id, args):
    loss_meter = AverageMeter()
    first_features, deep_features = [], []
    model.train()

    print(f"Starting epoch {epoch}, total batches: {len(loader)}")

    model_device = next(model.parameters()).device
    if model_device.type != 'cuda':
        print(f"WARNING: Model is not on GPU! device: {model_device}")

    for batch_idx, (image, label) in enumerate(loader):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()

        if batch_idx == 0:
            print(f"Epoch {epoch} - model:{model_device}, image:{image.device}, label:{label.device}")

        if epoch > 10:
            step_optimization(model, optimizer, criterion, image, label, loss_meter)
            _, _, mean_list, std_list = read_epoch_mean_std(style_folder, center_id, epoch)
            mean_img, std_img = GetMeanStd(image)
            for m, s in zip(mean_list, std_list):
                aug_img = ((image - mean_img) / (std_img + 1e-8)) * m + s
                aug_img = HalfMask(image, aug_img, image.size(-1), image.size(0), image.size(1)).cuda()
                style_mean, style_std = read_deep_mean_std(style_folder, epoch)
                step_optimization(model, optimizer, criterion, aug_img, label, loss_meter)
        else:
            step_optimization(model, optimizer, criterion, image, label, loss_meter)

        # 记录特征统计
        first_out = model.get_first_layer_output(image)
        mean, std = GetMeanStd(first_out)
        first_features.append((mean.detach().cpu().numpy(), std.detach().cpu().numpy()))
        deep_out = model.get_deep_layer_output(image)
        d_mean, d_std = GetMeanStd(deep_out)
        deep_features.append((d_mean.detach().cpu().numpy(), d_std.detach().cpu().numpy()))

    np.savetxt(f"{style_folder}C{center_id}_epoch{epoch}.csv", np.mean(first_features, axis=0), delimiter=",")
    np.savetxt(f"{style_folder}C{center_id}_epoch{epoch}_deep.csv", np.mean(deep_features, axis=0), delimiter=",")
    return loss_meter.avg

def Train_net(net, args, model_name, logger, step=0, per_epoch=5, center_id=0):
    checkpoint_dir_c = getattr(args, f'checkpoint_dir_c{center_id}', args.checkpoint_dir_c)
    model_path = os.path.join(checkpoint_dir_c, model_name)

    if not args.if_retrain and os.path.exists(model_path):
        try:
            net.load_state_dict(torch.load(model_path))
            print(f"Loaded checkpoint from {model_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}\nStart fresh.")
    elif args.if_retrain and os.path.exists(model_path) and step == 0:
        print(f"Checkpoint exists but --if_retrain=True, starting fresh training")

    if torch.cuda.is_available():
        net = net.cuda()
        print(f"CUDA OK. Model -> {next(net.parameters()).device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Training on CPU.")

    train_loader = DataLoader(Dataloader(args, center_id=center_id), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95))

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=5,      
    )

    criterion = cross_loss()

    logger.info(f'Start training step {step}!')
    for epoch in range(step * per_epoch, (step + 1) * per_epoch):

        loss = train_epoch(net, train_loader, optimizer, criterion, epoch, args.style_folder, center_id, args)

        # [MOD] 每个 epoch 后推进调度器
        scheduler.step(loss)

        print(f"Epoch {epoch}: Loss = {loss:.5f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # 存权重（先转 CPU 再存，避免 CUDA OOM）
        if torch.cuda.is_available():
            net_cpu = net.cpu()
            torch.save(net_cpu.state_dict(), model_path)
            net.cuda()
        else:
            torch.save(net.state_dict(), model_path)
        logger.info(f'Epoch:[{epoch}/{(step + 1) * per_epoch}] lr={optimizer.param_groups[0]['lr']:.6f} loss={loss:.5f}')

def read_file_from_txt(txt_path, base_dir=None):
    if base_dir is None:
        with open(txt_path, 'r') as f:
            return [line.strip() for line in f]
    else:
        if not os.path.isabs(txt_path):
            txt_path = os.path.join(base_dir, txt_path)
        with open(txt_path, 'r') as f:
            return [line.strip() for line in f]

def reshape_img(image, y, x):
    out = np.zeros([3, y, x], dtype=np.float32)
    out[:, :image.shape[1], :image.shape[2]] = image
    return out

# # [MOD] 如果完全不用加权核，可以不需要该函数；保留以兼容旧代码
# def create_map_kernel(shape):
#     # 改为常数 1 的核，等价于不加权融合
#     return np.ones((1,) + shape, dtype=np.float32)  # [MOD]

def create_map_kernel(shape):
    h, w = shape
    yy = np.arange(h) - h // 2
    xx = np.arange(w) - w // 2
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    k = 1.0 / (np.abs(Y)**4 + np.abs(X)**4 + 1e-8)
    return k[np.newaxis, :, :].astype(np.float32)


def predict(model, image_dir, save_path, args):
    print('Predicting test data')
    model.eval()
    with torch.no_grad():
        model_device = next(model.parameters()).device
        files = read_file_from_txt(image_dir, getattr(args, 'base_dir', None))

        mean, std = np.load(args.Meanstd_dir)
        mean, std = mean.reshape(3, 1, 1), std.reshape(3, 1, 1)
        shape = (args.ROI_shape, args.ROI_shape)
        kernel = create_map_kernel(shape)
        stride_x, stride_y = shape[0] // 2, shape[1] // 2  # 50% overlap

        for image_path in files:
            if getattr(args, 'base_dir', None) is not None and not os.path.isabs(image_path):
                image_path = os.path.join(args.base_dir, image_path)

            image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32).transpose(2, 0, 1)
            image = (image - mean) / std
            _, y, x = image.shape
            ind = ((max(y, x) // args.ROI_shape) + 1) * args.ROI_shape
            image = reshape_img(image, ind, ind)[np.newaxis, :, :, :]

            predict_arr = np.zeros([1, args.n_classes, ind, ind], dtype=np.float32)
            n_map = np.zeros_like(predict_arr)

            for i in range(0, ind - shape[0] + 1, stride_x):
                for j in range(0, ind - shape[1] + 1, stride_y):
                    crop = image[:, :, i:i + shape[0], j:j + shape[1]]
                    crop_tensor = torch.from_numpy(crop).to(model_device)
                    # output = model(crop_tensor).detach().cpu().numpy()  # no extra sigmoid
                    output = torch.sigmoid(model(crop_tensor)).detach().cpu().numpy() # [MOD]
                    predict_arr[:, :, i:i + shape[0], j:j + shape[1]] += output * kernel
                    n_map[:, :, i:i + shape[0], j:j + shape[1]] += kernel

            output = (predict_arr / np.maximum(n_map, 1e-8)).astype(np.float32)
            output_final = np.zeros([1, args.n_classes, y, x], dtype=np.float32)
            output_final[:, :, :y, :x] = output[:, :, :y, :x]

            out_img = Image.fromarray((255 * output_final[0, 0]).astype(np.uint8))
            if out_img.mode != 'L':
                out_img = out_img.convert('L')
            ensure_dir(save_path)
            out_img.save(os.path.join(save_path, os.path.basename(image_path)))
    print('Prediction complete!')

# def Dice(label_dir, pred_dir, base_dir=None):
#     label_file = read_file_from_txt(label_dir)
#     dice = np.zeros(len(label_file), dtype=np.float32)
#     for i, p in enumerate(label_file):
#         name = p[p.rfind('/') + 1:]

#         pred = np.array(Image.open(os.path.join(pred_dir, name)).convert('L'), dtype=np.float32) / 255.0
#         pred = (pred > 0.5).astype(np.float32)  # threshold 0.5

#         gt = np.array(Image.open(p).convert('L'), dtype=np.float32)
#         gt = (gt > 0).astype(np.float32)

#         inter = (pred * gt).sum()
#         dice[i] = (2.0 * inter) / (pred.sum() + gt.sum() + 1e-8)
#         print(name, dice[i])
#     return dice

def Dice(label_txt, pred_dir, base_dir=None):
    files = read_file_from_txt(label_txt) if base_dir is None else read_file_from_txt(label_txt, base_dir)
    scores = np.zeros(len(files), dtype=np.float32)
    for i, p in enumerate(files):
        gt_path = p if base_dir is None or os.path.isabs(p) else os.path.join(base_dir, p)
        name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, name)

        pred = np.array(Image.open(pred_path).convert('L'), dtype=np.float32)
        pred = pred / 255.0 if pred.max() > 1.0 else pred
        pred = (pred > 0.5).astype(np.float32)

        gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
        gt = (gt > 0).astype(np.float32)

        inter = (pred * gt).sum(dtype=np.float64)
        denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
        scores[i] = (2.0 * inter) / (denom + 1e-8)
        print(name, float(scores[i]))
    return scores

# def Dice(label_dir, pred_dir, base_dir=None):
#     def _files(txt_path, base):
#         if base is None:
#             return read_file_from_txt(txt_path)
#         else:
#             return read_file_from_txt(txt_path, base)

#     label_files = _files(label_dir, base_dir)
#     dice_scores = []
#     for file in label_files:
#         if base_dir is not None and not os.path.isabs(file):
#             file = os.path.join(base_dir, file)
#         name = os.path.basename(file)
#         pred = np.array(Image.open(os.path.join(pred_dir, name)).convert('L')).astype(np.float32)
#         pred = pred / 255.0
#         pred = np.where(pred > 0.5, 1, 0)
#         gt = np.array(Image.open(file).convert('L')).astype(np.float32)
#         gt = gt[0] if gt.ndim == 3 and gt.shape[0] == 3 else gt
#         gt = np.where(gt > 0, 1, 0)
#         inter = np.logical_and(pred, gt).sum()
#         dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-8)
#         print(name, dice)
#         dice_scores.append(dice)
#     return np.array(dice_scores)

def Create_files(args):
    for attr in ['checkpoint_dir', 'log_dir', 'checkpoint_dir_s', 'checkpoint_dir_c0', 'checkpoint_dir_c1',
                 'checkpoint_dir_c2', 'save', 'save_path', 'save_path_max', 'save_path_array', 'style_folder']:
        ensure_dir(getattr(args, attr, None))
    if hasattr(args, 'checkpoint_dir'):
        ensure_dir(os.path.join(args.checkpoint_dir, 'Server'))
        for i in range(3):
            ensure_dir(os.path.join(args.checkpoint_dir, f'C{i}'))
    if hasattr(args, 'style_folder'):
        ensure_dir(args.style_folder)

def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
