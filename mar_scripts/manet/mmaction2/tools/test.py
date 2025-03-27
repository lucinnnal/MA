# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
from unittest import result
import warnings
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from mmcv import Config, DictAction,mkdir_or_exist
import numpy as np
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        default=["top_k_accuracy"],
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    model.eval()

    # dataset & dataloader
    test_dataset = build_dataset(cfg.data.test)
    test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle = False)

    # model load
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    model.to(device)

    print("Starting evaluation on test dataset...")
    test_loss = 0.0
    correct = 0
    total = 0
    # loss
    criterion = nn.BCELoss()

    with torch.no_grad():
        for imgs, target in test_dataloader:  
            imgs, target = imgs.to(device), target.to(device)  
            
            target = target.float()

            output = model(imgs)

            loss = criterion(output, target)
            test_loss += loss.item()

            preds = (output > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.shape[0]

    # avg loss & metric
    avg_loss = test_loss / len(test_dataloader)
    accuracy = correct / total * 100

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()