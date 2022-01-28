import os
import argparse
from yacs.config import CfgNode
from .defaults import get_cfg_defaults


def load_cfg(args: argparse.Namespace):
    """Load configurations.
    """
    # Set configurations
    cfg = get_cfg_defaults()
    if args.config_base is not None:
        cfg.merge_from_file(args.config_base)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    overwrite_cfg(cfg, args)
    cfg.freeze()
    return cfg


def overwrite_cfg(cfg: CfgNode, args: argparse.Namespace):
    r"""Overwrite some configs given configs or args with higher priority.
    """
    # Distributed training:
    if args.distributed:
        cfg.SYSTEM.DISTRIBUTED = True
        cfg.SYSTEM.PARALLEL = 'DDP'

    if cfg.SOLVER.TEST_ONLY:
        cfg.DATASET.DATA_EXT = 'img'
