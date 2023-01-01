# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import copy
import torch
import models_mae_shared
import os.path
import numpy as np
from scipy import stats
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
import glob


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output is (B, classes)
    # target is (B)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_prameters_from_args(model, args):
    if args.finetune_mode == 'encoder':
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == 'all':
        parameters = model.parameters()
    elif args.finetune_mode == 'encoder_no_cls_no_msk':
        for name, p in model.named_parameters():
            if name.startswith('decoder') or name == 'cls_token' or name == 'mask_token':
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    return parameters


def _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device):
    if args.stored_latents:
        # We don't need to change the model, as it is never changed
        base_model.train(True)
        base_model.to(device)
        return base_model, base_optimizer, base_scalar
    clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    clone_model.train(True)
    clone_model.to(device)
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr, momentum=args.optimizer_momentum)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    else:
        assert args.optimizer_type == 'adam_w'
        optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    if args.load_loss_scalar:
        loss_scaler.load_state_dict(base_scalar.state_dict())
    return clone_model, optimizer, loss_scaler


def train_on_test(base_model: torch.nn.Module,
                  base_optimizer,
                  base_scalar,
                  dataset_train, dataset_val,
                  device: torch.device,
                  log_writer=None,
                  args=None,
                  num_classes: int = 1000, 
                  iter_start: int = 0,
                  reint = True):
    if args.model == 'mae_vit_small_patch16':
        classifier_depth = 8
        classifier_embed_dim = 512
        classifier_num_heads = 16
    else: 
        assert ('mae_vit_huge_patch14' in args.model or args.model == 'mae_vit_large_patch16') 
        classifier_embed_dim = 768
        classifier_depth = 12
        classifier_num_heads = 12
    clone_model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type, 
                                                         norm_pix_loss=args.norm_pix_loss, 
                                                         classifier_depth=classifier_depth, classifier_embed_dim=classifier_embed_dim, 
                                                         classifier_num_heads=classifier_num_heads,
                                                         rotation_prediction=False)
    # Intialize the model for the current run
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    train_loader = iter(torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers))
    
    accum_iter = args.accum_iter
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    dataset_len = len(dataset_val)
    dataset_len = 1001
    all_results = []
    all_losses =  [list() for i in range(dataset_len)]
    for data_iter_step in range(0, dataset_len):
        print(data_iter_step)
        if data_iter_step == dataset_len:
            break
        model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device)
        model.train()
        samples, _ = next(train_loader)
        samples = samples.to(device, non_blocking=True)[0] # index [0] becuase the data is batched to have size 1.
        loss_dict, _, _, _ = model(samples, None, mask_ratio=args.mask_ratio)

        loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters())

        optimizer.zero_grad()
                    
        metric_logger.update(**{k:v.item() for k,v in loss_dict.items()})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        
        with torch.no_grad():
            val_loader = iter(torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers))
            model.eval()
            all_pred = []
            for i in range(0, dataset_len):
                val_data = next(val_loader)
                (test_samples, test_label) = val_data
                test_samples = test_samples.to(device, non_blocking=True)[0]
                test_label = test_label.to(device, non_blocking=True)
                loss_d, _, _, pred = model(test_samples, test_label, mask_ratio=0, reconstruct=False)

                all_pred.extend(list(pred.argmax(axis=1).detach().cpu().numpy()))
                acc1 = (stats.mode(all_pred).mode[0] == test_label[0].cpu().detach().numpy()) * 100.
                all_results.append(acc1)
            
            with open(os.path.join(args.output_dir, 'results.npy'), 'ab') as f:
                np.save(f, np.array(all_results))
            print(len(all_results))
            all_results = []
    return 


def save_accuracy_results(args, model, save_model = False):
    all_all_results = [list() for i in range(args.steps_per_example)]
    for file_number, f_name in enumerate(glob.glob(os.path.join(args.output_dir, 'results_*.npy'))):
        all_data = np.load(f_name)
        for step in range(args.steps_per_example):
            all_all_results[step] += all_data[step].tolist()

    # for file_number, f_name in enumerate(glob.glob(os.path.join(args.output_dir, 'results_*.npy'))):
    #     os.remove(f_name)
    if save_model:
        torch.save(model, os.path.join(args.output_dir, 'model-final.pth'))
    else:
        with open(os.path.join(args.output_dir, 'model-final.pth'), 'w') as f:
            f.write(f'Done!\n')
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        for i in range(args.steps_per_example):
            assert len(all_all_results[i]) == 1001, len(all_all_results[i])
            f.write(f'{i}\t{np.mean(all_all_results[i])}\n')