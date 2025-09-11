# -*- coding: utf-8 -*-


import torch

from torch import nn
from torch.nn.functional import max_pool3d


class dice_coef(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1.
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        a = torch.sum(y_true * y_pred, (2, 3, 4))
        b = torch.sum(y_true, (2, 3, 4))
        c = torch.sum(y_pred, (2, 3, 4))
        dice = (2 * a) / (b + c + smooth)
        return torch.mean(dice)

class mix_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):

        return crossentry()(y_true, y_pred) + 1 - dice_coef()(y_true, y_pred)

class w_dice_ce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):

        return 0.05*(1-dice_coef()(y_true, y_pred))+w_crossentry()(y_true, y_pred)


class crossentry(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred+smooth))
    
class cross_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred+smooth)+ (1-y_true) * torch.log(1 - y_pred+smooth))
    
class cross_loss_stenosis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, stenosis):
        smooth = 1e-6
        return -torch.sum(stenosis*(y_true * torch.log(y_pred+smooth)+ (1-y_true) * torch.log(1 - y_pred+smooth)))/(torch.sum(stenosis)+smooth)

class w_crossentry(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=1.8):
        smooth = 1e-6
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        w = y_true - y_pred
        a = torch.pow(w, 2) * ((alpha * y_true * torch.log(y_pred+smooth)+(2-alpha) * (1 - y_true)*torch.log(1-y_pred+smooth)))
        return -torch.mean(a)

class cut_crossentry(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        print(torch.sum(w).data)

        loss_ce = -torch.sum(w *(y_true * torch.log(y_pred + smooth) +
                                  w * (1 - y_true) * torch.log(1 - y_pred + smooth))) / torch.sum(w + smooth)

        return loss_ce

class cut_crossentry_line(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_mask, y_pred, alpha=0.45):
        smooth = 1e-6
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        y_mask = max_pool3d(y_mask, size, size)

        w = torch.abs(y_true - y_pred)
        w_low = torch.round(w + alpha)
        w_high = 1 - torch.round(w - alpha)
        w = w_low * w_high
        print(torch.sum(w).data)
        loss_ce = -torch.sum(w *(y_true * torch.log(y_pred + smooth) +
                                 (1-y_mask)*(1 - y_true) * torch.log(1 - y_pred + smooth))) / torch.sum(w + smooth)
        return loss_ce

class balance_bce_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = -((torch.sum(w * y_true * torch.log(y_pred + smooth)) / torch.sum(w * y_true+ smooth)) +
                             (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) / torch.sum(w * (1 - y_true) + smooth)))/2
        
        #temp_y = torch.where(y_pred>=0, torch.full_like(y_pred, 1), torch.full_like(y_pred, 0))
        #print(torch.sum(w)/torch.sum(temp_y))
        return loss_ce
    
class balance_bce_loss_softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        y_pred = y_pred[:, 1, :, :, :]
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = -((torch.sum(w * y_true * torch.log(y_pred + smooth)) / torch.sum(w * y_true+ smooth)) +
                             (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) / torch.sum(w * (1 - y_true) + smooth)))/2
        

        return loss_ce

class cut_crossentry_softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, label, alpha=0.45):
        smooth = 1e-6

        w = torch.abs(label - predict)
        w = torch.round(w + alpha)
        print(torch.sum(w).data)

        loss_ce = -torch.sum(w * label * torch.log(predict + smooth)) / torch.sum(w + smooth)
        return loss_ce

class crossentry_centerline(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, label, alpha=0.45):
        smooth = 1e-6

        w = torch.abs(label - predict)
        w = torch.round(w + alpha)
        print(torch.sum(w).data)

        loss_ce = -torch.sum(w * label * torch.log(predict + smooth)) / torch.sum(w + smooth)
        return loss_ce
    
class stenosis_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, label_line, predict):
        smooth = 1e-6
        label_line_sum=torch.sum(label_line)
        #predict=torch.where(predict > 0.5, torch.full_like(predict, 1), torch.full_like(predict, 0))
        label_line_in=predict*label_line

        label_line_in_sum=torch.sum(label_line_in)

        IoU=(label_line_in_sum+smooth)/(label_line_sum+smooth)
        return -torch.log(IoU)

class w_stenosis_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, label, label_line, predict, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(label - predict)
        w = torch.round(w + alpha)
        
        loss_ce = -((torch.sum(w * label * torch.log(predict + smooth)) / torch.sum(w * label+ smooth)) +
                             (torch.sum(w * (1 - label) * torch.log(1 - predict + smooth)) / torch.sum(w * (1 - label) + smooth)))/2       
        
        w_total1 = torch.mean(w)
        w_total1 = torch.abs(w_total1)
        w_total2 = torch.mean(1000 - 1000 * w)
        w_total2 = torch.abs(w_total2)
        print(w_total2.data)
        
        
        label_line_sum=torch.sum(label_line)
        predict=torch.where(predict > 0.5, torch.full_like(predict, 1), torch.full_like(predict, 0))
        label_line_in=predict*label_line

        label_line_in_sum=torch.sum(label_line_in)

        IoU=(label_line_in_sum+smooth)/(label_line_sum+smooth)
        
        return  w_total1*loss_ce - w_total2*torch.log(IoU)

class balance_bce_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = -((torch.sum(w * y_true * torch.log(y_pred + smooth)) / torch.sum(w+ smooth)) +
                             (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) / torch.sum(w + smooth)))/2
        return loss_ce
    
class balance_dice_coef(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1.
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        a = torch.sum(w * y_true * w * y_pred, (2, 3, 4))
        b = torch.sum(w * y_true, (2, 3, 4))
        c = torch.sum(w * y_pred, (2, 3, 4))
        dice = (2 * a) / (b + c + smooth)
        loss = torch.mean(dice)
        return loss
