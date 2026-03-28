"""
Training script for MoKGR on RecSys datasets.

Usage
-----
    python train.py --data_path data/last-fm/ --gpu 0
    python train.py --data_path data/amazon-book/ --gpu 0
    python train.py --data_path data/alibaba-fashion/ --gpu 0
"""

import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import checkPath

parser = argparse.ArgumentParser(description="MoKGR for Recommendation")
parser.add_argument('--data_path', type=str, default='data/last-fm/')
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epoch', type=int, default=40)
# gate
parser.add_argument('--gate_threshold', type=float, default=0.1)
parser.add_argument('--active_gate', action='store_true')
# PPR
parser.add_argument('--sampling_percentage', type=float, default=0.85)
parser.add_argument('--PPR_alpha', type=float, default=0.85)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--active_PPR', action="store_true")
# MoE for hops
parser.add_argument('--num_experts', type=int, default=3)
parser.add_argument('--min_hop', type=int, default=2)
parser.add_argument('--max_hop', type=int, default=5)
parser.add_argument('--lambda_importance', type=float, default=1e-7)
parser.add_argument('--lambda_load', type=float, default=0)
parser.add_argument('--lambda_noise', type=float, default=1)
parser.add_argument('--hop_temperature', type=float, default=1.1)
# MoE for pruning
parser.add_argument('--pruning_temperature', type=float, default=1.5)
parser.add_argument('--K_source', type=int, default=1000)
parser.add_argument('--K_min', type=int, default=1000)
parser.add_argument('--K_max', type=int, default=2000)
parser.add_argument('--l_inflection', type=int, default=3)
parser.add_argument('--a', type=float, default=3.0)
parser.add_argument('--num_pruning_experts', type=int, default=2)
parser.add_argument('--lambda_importance_pruning', type=float, default=1e-7)
parser.add_argument('--lambda_noise_pruning', type=float, default=1.0)
# logging
parser.add_argument('--log_file', type=str, default='train.log')
args = parser.parse_args()


if __name__ == '__main__':
    args.n_layer = args.max_hop

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path.rstrip('/').split('/')[-1]

    checkPath('./results/')
    opts = args

    torch.cuda.set_device(opts.gpu)
    print(f'==> gpu: {opts.gpu}')

    # ---- data ----
    loader = DataLoader(
        args.data_path,
        active_PPR=args.active_PPR,
        sampling_percentage=args.sampling_percentage,
        PPR_alpha=args.PPR_alpha,
        max_iter=args.max_iter
    )
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    opts.n_users = loader.n_users
    opts.n_items = loader.n_items
    opts.n_nodes = loader.n_nodes

    # ---------------------------------------------------------------
    # Per-dataset hyper-parameter presets
    # ---------------------------------------------------------------
    if dataset == 'last-fm':
        opts.lr         = 0.0004
        opts.decay_rate = 0.994
        opts.lamb       = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 30
        opts.n_tbatch   = 30

    elif dataset == 'new_last-fm':
        opts.lr         = 0.0004
        opts.decay_rate = 0.994
        opts.lamb       = 0.00014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 36
        opts.n_tbatch   = 36

    elif dataset == 'amazon-book':
        opts.lr         = 0.0012
        opts.decay_rate = 0.994
        opts.lamb       = 0.000014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 20
        opts.n_tbatch   = 20

    elif dataset == 'new_amazon-book':
        opts.lr         = 0.0005
        opts.decay_rate = 0.994
        opts.lamb       = 0.000014
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 24
        opts.n_tbatch   = 24

    elif dataset == 'alibaba-fashion':
        opts.lr         = 10 ** -6.5
        opts.decay_rate = 0.998
        opts.lamb       = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.2
        opts.act        = 'relu'
        opts.n_batch    = 10
        opts.n_tbatch   = 10

    elif dataset == 'new_alibaba-fashion':
        opts.lr         = 0.00005
        opts.decay_rate = 0.999
        opts.lamb       = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 20
        opts.n_tbatch   = 20

    elif dataset == 'Dis_5fold_item':
        opts.lr         = 0.0005
        opts.decay_rate = 0.994
        opts.lamb       = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 20
        opts.n_tbatch   = 20

    elif dataset == 'Dis_5fold_user':
        opts.lr         = 0.001
        opts.decay_rate = 0.994
        opts.lamb       = 0.00001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.01
        opts.act        = 'idd'
        opts.n_batch    = 24
        opts.n_tbatch   = 24

    else:
        # sensible fallback
        opts.lr         = 0.0002
        opts.decay_rate = 0.9938
        opts.lamb       = 0.0001
        opts.hidden_dim = 48
        opts.attn_dim   = 5
        opts.n_layer    = args.max_hop
        opts.dropout    = 0.02
        opts.act        = 'idd'
        opts.n_batch    = 20
        opts.n_tbatch   = 20

    opts.perf_file = os.path.join('results', f'{dataset}_perf.txt')

    config_str = (f'lr={opts.lr:.6f}, decay={opts.decay_rate:.4f}, lamb={opts.lamb:.6f}, '
                  f'dim={opts.hidden_dim}, attn={opts.attn_dim}, layers={opts.n_layer}, '
                  f'batch={opts.n_batch}, drop={opts.dropout:.4f}, act={opts.act}, '
                  f'max_hop={opts.max_hop}, min_hop={opts.min_hop}, '
                  f'num_experts={opts.num_experts}, num_pruning_experts={opts.num_pruning_experts}\n')
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    # ---- model ----
    model = BaseModel(opts, loader)

    best_recall = 0.0
    best_str = ''
    for epoch in range(args.epoch):
        print(f'\n===== epoch {epoch} =====')
        recall, ndcg, out_str = model.train_batch()

        with open(opts.perf_file, 'a+') as f:
            f.write(f'epoch {epoch}  {out_str}')

        if recall > best_recall:
            best_recall = recall
            best_str = out_str
            print(f'  *** new best @ epoch {epoch}: {best_str.strip()}')

    with open(opts.perf_file, 'a+') as f:
        f.write(f'best:\n{best_str}')
    print(f'\n==> Best result: {best_str}')
