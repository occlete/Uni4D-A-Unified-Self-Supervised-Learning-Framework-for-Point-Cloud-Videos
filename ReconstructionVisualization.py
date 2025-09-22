from __future__ import print_function
import datetime
import os
import time
import sys
import random
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.utils.data
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import utils
from logger import setup_logger
from data_aug.CLR_MSR import ContrastiveLearningDataset
#from models.CLR_Model import ContrastiveLearningModel
#from models.CLR_SFFormer import ContrastiveLearningModel
from timm.scheduler import CosineLRScheduler
#from models.CMAE_PCS_EMA import ContrastiveMaskedAutoEncoder
from models.Uni4DMSR import ContrastiveMaskedAutoEncoder
#from data_aug.CMAE_MSRV2 import MSRAction3D

def train(model, optimizer, lr_scheduler, data_loader, 
        device, epoch, print_freq, logger):

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    
    model.train()
    for i, (clips, video_index) in enumerate(data_loader):
        start_time = time.time()
        clips = clips.to(device)

        loss = model(clips)
        loss = loss.mean()

        batch_size = clips.shape[0]
        lr_ = optimizer.param_groups[-1]["lr"]

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start_time)

        if i % print_freq == 0:
            logger.info(('Epoch:[{0}][{1}/{2}]\t'
                         'lr:{lr:.5f}\t'
                         'Loss:{loss.val:.6f} ({loss.avg:.6f})\t'.format(
                            epoch, i, len(data_loader), 
                            lr=lr_, 
                            loss=losses))) 

        if isinstance(lr_scheduler, list):
            for item in lr_scheduler:
                item.step(epoch)
        else:
            lr_scheduler.step(epoch)

    return losses.avg

def evalate(model, data_loader, device):

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
    #model.train()
        for i, (clips,_) in enumerate(data_loader):
            start_time = time.time()
            #clips = clips.to(device)
            clips = clips.to(device)
            #clips_full = clips_full.to(device)
            pre_points, gt_points, vis_points,mask= model(clips)
            """
            pre_points(b l*masked tk knn 3)
            gt_points(b l*masked tk knn 3)
            vis_pooints(b l*unmasked tk knn 3)
            """
            clips = clips.cpu().numpy()
            pre_points = pre_points.cpu().numpy()
            gt_points = gt_points.cpu().numpy()
            vis_points = vis_points.cpu().numpy()
            #vis_center=vis_center.cpu().numpy()
            #x_denoise_np = x_denoise_np * scale + shift
            # 保存为npy文件
            clips_path = os.path.join('/home/pod/shared-nvme/Vis_msr', f'clips_{i}.npy')
            pre_points_path = os.path.join('/home/pod/shared-nvme/Vis_msr', f'pre_points_{i}.npy')
            gt_points_path = os.path.join('/home/pod/shared-nvme/Vis_msr', f'gt_points_{i}.npy')
            vis_points_path = os.path.join('/home/pod/shared-nvme/Vis_msr', f'vis_points_{i}.npy')
            #vis_center_path = os.path.join('/mnt/hdd/ZuoZhi/code/MaST-Pre-main/Vis', f'vis_center_{i}.npy')
            np.save(clips_path,clips)
            np.save(pre_points_path, pre_points)
            np.save(gt_points_path, gt_points)
            np.save(vis_points_path, vis_points)
            #np.save(vis_center_path, vis_center)

        


def main(args):

    # Fix the seed 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    # Check folders and setup logger
    log_dir = os.path.join(args.log_dir, args.model)
    utils.mkdir(log_dir)

    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    logger = setup_logger(output=log_dir, distributed_rank=0, name=args.model)
    tf_writer = SummaryWriter(log_dir=log_dir)
    
    # Data loading code
    train_dataset = ContrastiveLearningDataset(
                    root=args.data_path,
                    #meta=args.data_meta,
                    frames_per_clip=args.clip_len,
                    step_between_clips=args.clip_stride,
                    #step_between_frames=args.frame_stride,
                    num_points=args.num_points,
                    train=True
    )
    """
    train_dataset = MSRAction3D()
    """
    train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    num_workers=args.workers, 
                    pin_memory=True, 
                    drop_last=True
    )
    # Creat Contrastive Learning Model
    model = ContrastiveMaskedAutoEncoder(
            radius=args.radius, 
            nsamples=args.nsamples, 
            spatial_stride=args.spatial_stride,
            temporal_kernel_size=args.temporal_kernel_size,
            temporal_stride=args.temporal_stride,
            en_emb_dim=args.en_dim, 
            en_depth=args.en_depth, 
            en_heads=args.en_heads, 
            en_head_dim=args.en_head_dim, 
            en_mlp_dim=args.en_mlp_dim,
            de_emb_dim=args.de_dim, 
            de_depth=args.de_depth, 
            de_heads=args.de_heads, 
            de_head_dim=args.de_head_dim, 
            de_mlp_dim=args.de_mlp_dim,
            mask_ratio=args.mask_ratio,
            dropout1=args.dropout1,
            pretraining=True,
            vis=True,
    )
    # Distributed model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)


    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    param_groups = add_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = CosineLRScheduler(optimizer,
            t_initial=args.epochs,
            #t_mul=1,
            lr_min=1e-6,
            #decay_rate=0.1,
            warmup_lr_init=1e-6,
            warmup_t=args.lr_warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("===> Loading checkpoint for resume '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            logger.info(("===> Loaded checkpoint with epoch {}".format(checkpoint['epoch'])))
        else:
            logger.info(("===> There is no checkpoint at '{}'".format(args.resume)))

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        #model.eval()
        evalate(model,train_loader,device)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch')

    #parser.add_argument('--data-path', default='/mnt/hdd/ZuoZhi/ntu', type=str, help='dataset')#NTU
    #parser.add_argument('--data-meta', default='/mnt/hdd/ZuoZhi/ntu60.list', help='dataset')#NTU
    parser.add_argument('--data-path', default='/home/pod/shared-nvme/ReSD/msr_action', type=str, help='dataset')
    parser.add_argument('--epochs', default=201, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    # input
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    #parser.add_argument('--clip-stride', default=8, type=int, metavar='N', help='number of steps between clips')#NTU
    parser.add_argument('--clip-stride', default=1, type=int, metavar='N', help='number of steps between clips')#MSR
    parser.add_argument('--frame-stride', default=2, type=int, metavar='N', help='number of steps between clips')
    parser.add_argument('--num-points', default=1024, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('--mask-ratio', default=0.75, type=float, metavar='N', help='mask ratio')
    # P4D
    parser.add_argument('--radius', default=0.3, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # encoder transformer
    parser.add_argument('--en-dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--en-depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--en-heads', default=8, type=int, help='transformer head')
    parser.add_argument('--en-head-dim', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--en-mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # decoder transformer
    parser.add_argument('--de-dim', default=512, type=int, help='transformer dim')
    parser.add_argument('--de-depth', default=4, type=int, help='transformer depth')
    parser.add_argument('--de-heads', default=8, type=int, help='transformer head')
    parser.add_argument('--de-head-dim', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--de-mlp-dim', default=1024, type=int, help='transformer mlp dim')
    parser.add_argument('--dropout1', default=0., type=float, help='transformer dropout')
    parser.add_argument('--dropout-cls', default=0., type=float, help='classifier dropout')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.05, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='/home/pod/shared-nvme/Pretrainckpt/9303/pretrain/checkpoint_149.pth', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=199, type=int, metavar='N', help='start epoch')
    parser.add_argument('--model', default='ContrastiveMaskedAutoEncoder', type=str, help='model')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    #parser.add_argument('--log-dir', default='/mnt/hdd/ZuoZhi/code/MaST-Pre-main/SFFormer_Next/', type=str, help='path where to save')
    parser.add_argument('--log-dir', default='/home/pod/shared-nvme/Vis_msr', type=str, help='path where to save')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
