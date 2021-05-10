#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import setup_logger
from AttaNet import AttaNet
from cityscapes import CityScapes
from loss import OhemCELoss
from evaluate import evaluate
from optimizer import Optimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

import os
import os.path as osp

import time
import logging
import datetime
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
respth = './res'
if not osp.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest='local_rank',
            type=int,
            default=-1,
            )
    parse.add_argument(
        '--ckpt',
        dest='ckpt',
        type=str,
        default=None,
    )
    parse.add_argument("--save-pred-every", type=int, default=2000,
                       help="Save summaries and checkpoint every often.")
    parse.add_argument("--snapshot-dir", type=str, default='./snapshots/',
                       help="Where to save snapshots of the model.")
    return parse.parse_args()


def train():
    args = parse_args()
    dist.init_process_group(
                backend='nccl',
                world_size=torch.cuda.device_count()
                )
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)                
    setup_logger(respth)

    # dataset
    n_classes = 19
    n_img_per_gpu = 8
    n_workers = 4
    cropsize = [1024, 1024]
    ds = CityScapes('../data/cityscapes', cropsize=cropsize, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    logger.info('successful load data')

    ignore_idx = 255
    net = AttaNet(n_classes=n_classes)
    if not args.ckpt is None:
        net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        logger.info('successful load weights')
    net.cuda(device)
    net.train()
    net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)
    logger.info('successful distributed')
    score_thres = 0.7
    n_min = cropsize[0]*cropsize[1]//2
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_aux1 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_aux2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 200000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model=net.module,
            lr0=lr_start,
            momentum=momentum,
            wd=weight_decay,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            max_iter=max_iter,
            power=power)

    # train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, out16, out32 = net(im)
        lossp = criteria_p(out, lb)
        loss1 = criteria_aux1(out16, lb)
        loss2 = criteria_aux2(out32, lb)
        loss = lossp + loss1 + loss2
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        # print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it=it+1,
                    max_it=max_iter,
                    lr=lr,
                    loss=loss_avg,
                    time=t_intv,
                    eta=eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed

    save_pth = osp.join(args.snapshot_dir, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    train()
    evaluate()
