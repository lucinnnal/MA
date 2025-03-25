# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import torch
from torch import nn
import os
import os.path as osp
import time
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import mmcv
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from mmcv import Config, DictAction,mkdir_or_exist
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--task_name', type=str, default='formal')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        default='True',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        default=1,
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        default=[0],
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        default='True',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    # 모델 가중치 경로
    checkpoint_path = '/Users/kipyokim/Desktop/Micro-Action/mar_scripts/manet/pretrained_weights.pth' 
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])
    print(model.cls_head.fc_cls)

    # model의 해드 부분 변경
    model.cls_head.fc_cls = nn.Linear(2048, 1)
    
    # 수정 후 모델
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Freezing
    for name, param in model.named_parameters():
        if "cls_head.fc_cls" in name:
            param.requires_grad = True  # fc_cls는 학습 가능
        else:
            param.requires_grad = False  # 나머지는 freeze
    print("freezed except the head classifier")

    # Freeze 잘 되었는지 확인
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    x = torch.randint(0, 256, (1, 8, 3, 224, 224), dtype=torch.uint8)
    x = x.float()
    x.to(device)
    result = model(x)
    print(result)

if __name__ == '__main__':
    main()