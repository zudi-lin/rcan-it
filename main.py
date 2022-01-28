import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from ptsr import model
from ptsr.data import Data
from ptsr.config import load_cfg
from ptsr.model import get_num_params
from ptsr.utils import utility, trainer


def init_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
    parser.add_argument('--config-file', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--manual-seed', type=int, default=None)
    parser.add_argument('--local_world_size', type=int, default=1,
                        help='number of GPUs each process.')
    parser.add_argument('--local_rank', type=int, default=None,
                        help='node rank for distributed training')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = load_cfg(args)

    if args.distributed:  # parameters to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                        "LOCAL_RANK", "WORLD_SIZE")}
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        n = torch.cuda.device_count() // args.local_world_size
        device_ids = list(
            range(args.local_rank * n, (args.local_rank + 1) * n))

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        print(
            f"[{os.getpid()}] rank = {dist.get_rank()} ({args.rank}), "
            + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )

        manual_seed = args.local_rank if args.manual_seed is None \
            else args.manual_seed
    else:
        manual_seed = 0 if args.manual_seed is None else args.manual_seed
        device = torch.device('cuda:0')

    # init random seeds for reproducibility
    init_seed(manual_seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    if args.local_rank == 0 or args.local_rank is None:
        print(cfg)

    # initialize model, loss and loader
    checkpoint = utility.checkpoint(cfg)
    _model, _loss = build_model_loss(cfg, args.local_rank, checkpoint, device)
    loader = Data(cfg)

    t = trainer.Trainer(cfg, args.local_rank, loader,
                        _model, _loss, device, checkpoint)
    checkpoint.load_model(
        pre_train=cfg.MODEL.PRE_TRAIN, trainer=t, device=device,
        restart=cfg.SOLVER.ITERATION_RESTART, test_mode=cfg.SOLVER.TEST_ONLY,
        strict=cfg.MODEL.CKP_STRICT, ignore=cfg.MODEL.CKP_IGNORE)
                          
    t.test() if cfg.SOLVER.TEST_ONLY else t.train()
    if args.distributed:
        dist.destroy_process_group()  # tear down the process group


def build_model_loss(cfg, rank, checkpoint, device):
    _model = model.Model(cfg, checkpoint).to(device)
    if rank is None or rank == 0:
        print("Total number of parameters: ", get_num_params(_model))

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    find_unused = cfg.MODEL.STOCHASTIC_DEPTH or (cfg.SOLVER.TAIL_ONLY_ITER > 0)
    if cfg.SYSTEM.PARALLEL == "DDP":
        _model = nn.parallel.DistributedDataParallel(
            _model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused)
    else:
        _model = nn.parallel.DataParallel(_model)  # parallel on all devices

    _loss = None
    if not cfg.SOLVER.TEST_ONLY:
        _loss = nn.L1Loss().to(device)

    return _model, _loss


if __name__ == '__main__':
    main()
