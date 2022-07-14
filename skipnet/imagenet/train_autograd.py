"""
Training file for HRL stage. Support Pytorch 3.0 and multiple GPUs.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import autograd
from torch.autograd import Variable
from torch import optim
import torch.nn.parallel
import torch.optim
from torch.optim.optimizer import Optimizer
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import os
import shutil
import argparse
import time
import logging
import json
import itertools
import models as models
import sys
import pdb

from utils import ListAverageMeter, AverageMeter, more_config, accuracy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet training with gating')
    parser.add_argument('--cmd', choices=['train', 'test'])
    parser.add_argument('--model-type', metavar='ARCH', default='rl', choices=['rl','sp','sp-50'])
    parser.add_argument('--gate-type', default='rnn',
                        choices=['rnn'], help='gate type,only support RNN Gate')
    parser.add_argument('--data', '-d', default='dataset/imagenet/',
                        type=str, help='path to the imagenet data')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum used in SGD')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained',
                        action='store_true', help='use pretrained model')
    parser.add_argument('--save-folder', default='save_checkpoints',
                        type=str, help='folder to save the checkpoints')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='cropping size of the input')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scaling size of the input')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--rl-weight', default=0.01, type=float,
                        help='scaling weight for rewards')
    parser.add_argument('--gamma', default=1, type=float,
                        help='discount factor')
    parser.add_argument('--restart', action='store_true', help='restart ckpt')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for gate parameter initialization')
    parser.add_argument('--K', type=float, default=1)
    parser.add_argument('--acc-maintain', action='store_true', help='to disturb the sample maintaining the accuracy')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_type == 'rl':
        args.arch = "imagenet_rnn_gate_rl_101"
        args.resume = "resnet-101-rnn-imagenet.pth.tar"
    elif args.model_type == 'sp':
        args.arch = "imagenet_rnn_gate_101"
        args.resume = "resnet-101-rnn-sp-imagenet.pth.tar"
    elif args.model_type == 'sp-50':
        args.arch = "imagenet_rnn_gate_50"
        args.resume = "resnet-50-rnn-sp-imagenet.pth.tar"
    more_config(args)

    print(args)
    logging.info('CMD: '+' '.join(sys.argv))

    test_model(args)

def test_model(args):
    # create model
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = transforms.Compose([
        transforms.Scale(args.scale_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, t),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(args, val_loader, model)

def tanh_rescale(x, x_min=[-2.4291,-2.4183,-2.2214], x_max=[2.5141,2.5968,2.7537], type=False):
    x_min, x_max = torch.tensor(x_min)[None,:,None,None], torch.tensor(x_max)[None,:,None,None]
    return (torch.tanh(0.8*x) + 1) / 2 * (x_max - x_min) + x_min

def gate_loss(logprobs, threshold=0.5, upper_bound=1.0):

    gateloss_pos = -torch.clamp(logprobs-threshold, min=0)
    gateloss_pos = gateloss_pos.sum()

    gateloss_neg = torch.clamp(threshold-logprobs, min=0)
    gateloss_neg = gateloss_neg.sum()

    return gateloss_pos, gateloss_neg

def validate(args, val_loader, model):
    batch_time = AverageMeter()
    skip_ori_ratios = ListAverageMeter()
    skip_ratios = ListAverageMeter()
    prec1s_ori = AverageMeter()
    prec1s_mod = AverageMeter()
    iters = AverageMeter()
    vars = AverageMeter()
    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = input

        # calculating and recording the original result
        output, masks, _, _  = model(input_var)
        prec1, = accuracy(output.data, target, topk=(1,))
        skips_ori = masks.detach().mean(0)
        skip_ori_ratios.update(skips_ori, input.size(0))
        prec1s_ori.update(prec1)

        # perturbation initializing
        modifier = torch.zeros(input_var.size()).float()
        modifier_var = autograd.Variable(modifier, requires_grad=True)
        optimizer = optim.Adam([modifier_var], lr=args.lr)

        # training
        for iter in range(300):
            skips, var, prec1 = optimize(optimizer, model, input_var, modifier_var, target, output.data, iter, args)

        # break
        # path = "./numpy_output/ours_" + args.model_type + "_k_{}".format(args.K)
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # input_adv = tanh_rescale(modifier_var + input_var)
        # samples = torch.stack([input_var, input_adv]).transpose(0,1).detach().cpu().numpy()
        # for sample in samples:
        #     numpy.save(path+"/ori_{:05d}".format(index_modified),sample[0])
        #     numpy.save(path+"/mod_{:05d}".format(index_modified),sample[1])
        #     index_modified = index_modified + 1

        # result recording
        skip_ratios.update(skips.mean(0), skips.shape[0])
        vars.update(var.mean(0), var.shape[0])
        prec1s_mod.update(prec1)
        iters.update(iter+1)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (i == (len(val_loader) - 1)):
            cp_ori = ((sum(skip_ori_ratios.avg) + 1) / (skip_ori_ratios.len + 1)) * 100
            cp = ((sum(skip_ratios.avg) + 1) / (skip_ratios.len + 1)) * 100
            logging.info('***Computation Percentage: from {:.3f}% to {:.3f}%'.format(cp_ori, cp))
            logging.info('***Original prec: from {:.5f} to {:.5f}'.format(prec1s_ori.avg, prec1s_mod.avg))
            logging.info('***Final Iter: {:.5f}'.format(iters.avg))
            logging.info('***Final Var: {:.5f}'.format(vars.avg))

    # always keep the first block
    cp = ((sum(skip_ori_ratios.avg) + 1) / (skip_ori_ratios.len + 1)) * 100
    logging.info('***Original Computation Percentage: {:.3f} %'.format(cp))

    cp = ((sum(skip_ratios.avg) + 1) / (skip_ratios.len + 1)) * 100
    logging.info('***Final Computation Percentage: {:.3f} %'.format(cp))

    logging.info('***Final Var: {:.5f}'.format(iters.avg))

def optimize(optimizer, model, input_var, modifier_var, target, ori_output, iter, args):

    # output computing
    input_adv = tanh_rescale(modifier_var + input_var)
    output, masks, gates, hidden = model(input_adv)
    
    if args.acc_maintain:
        l2_dist = ((input_adv-input_var)**2).sum() + ((ori_output-output)**2).sum()*100000
    else:
        l2_dist = ((input_adv-input_var)**2).sum()
    gateloss_pos, gateloss_neg = gate_loss(gates, 0.5)
    
    # L2 gradient calculating
    optimizer.zero_grad()
    l2_dist.backward(retain_graph=True)
    l2_dist_grad = modifier_var.grad.clone().detach()

    # postive gradient calculating
    optimizer.zero_grad()
    gateloss_pos.backward(retain_graph=True)
    gradient_pos = modifier_var.grad.clone().detach()

    # negative gradient calculating
    optimizer.zero_grad()
    gateloss_neg.backward()
    gradient_neg = modifier_var.grad.clone().detach()

    # constrain calculating
    mag_pos = gradient_pos.norm(p=2, dim=[1,2,3])
    direction = gradient_pos/(mag_pos[:,None,None,None]+1e-20)
    projection = torch.einsum('nijk,nijk->n', gradient_neg, direction)
    projection[projection>0] = 0
    gradient_neg -= torch.einsum('n,nijk->nijk',projection, direction)
    mag_neg = gradient_neg.norm(p=2, dim=[1,2,3])
    gradient_neg /= (mag_neg[:,None,None,None]+1e-20)

    # optimizing
    shape = input_var.shape[1:].numel()
    modifier_var.grad = args.K * shape * gradient_neg + l2_dist_grad
    optimizer.step()

    # result calculating
    img_size = input_var.shape[1] * input_var.shape[2] * input_var.shape[3]
    vars = ((input_adv-input_var)**2).sum([1,2,3])/img_size
    prec1, = accuracy(output.data, target, topk=(1,))

    # removing the sample with nan pixels
    index = vars!=torch.nan
    skips = masks[index].detach()
    vars = vars[index].detach()

    return skips, vars, prec1

if __name__ == '__main__':
    main()