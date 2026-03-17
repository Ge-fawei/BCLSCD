import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import datetime
import shutil

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity, CrossEntropy2d
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

###############################################
from dataset_process import SECOND_process as SP
from models.BCL import BCL as Net

NET_NAME = 'BCL'
DATA_NAME = 'SECOND'
###############################################

###############################################
args = {
    'train_batch_size': 8,
    'val_batch_size': 8,
    'lr': 0.06,
    'epochs': 50,
    'gpu': True,
    'gpu_ids': 0,
    'lr_decay_power': 1.5,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 50,
    'predict_step': 5,
    'NET_NAME': 'SCD',
    'DATA_NAME': 'SECOND',
    'name': './model/SECOND/ours/TBFFNet_68e_mIoU73.95_Sek24.48_Fscd63.86_OA88.24.pth',
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'trained_model', DATA_NAME, 'pretrained.pth')
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
writer = SummaryWriter(args['log_dir'])


gpu_ids = []
gpu_ids.append(args['gpu_ids'])


# set gpu ids
if len(gpu_ids)>1:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    cudnn.enabled = True
    cudnn.benchmark = True
else:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True



def main():
    net = Net(3, SP.num_classes).cuda()
    net.load_state_dict(torch.load(args['name']))

    train_set = SP.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = SP.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    train(train_loader, net, criterion, optimizer, scheduler, val_loader)
    writer.close()



def train(train_loader, net, criterion, optimizer, scheduler, val_loader):
    
        curr_epoch = 50
        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        





def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, SP.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()