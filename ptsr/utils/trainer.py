import os
import math
import GPUtil
from decimal import Decimal

import torch
import torch.nn.utils as utils
from . import utility
from .solver import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from .solver.build import build_swa_model
from ..data.common import Mixup

class Trainer():
    def __init__(self, cfg, rank, loader, model, loss, device, ckp=None):
        self.cfg = cfg
        self.rank = rank
        self.scale = cfg.DATASET.DATA_SCALE

        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.device = device
        self.iter_start = 0

        if not cfg.SOLVER.TEST_ONLY:
            self.loader_train = iter(loader.loader_train)
            self.optimizer = build_optimizer(cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
            self.iteration_total = self.cfg.SOLVER.ITERATION_TOTAL
            self.tail_only_iter = self.cfg.SOLVER.TAIL_ONLY_ITER
            self.best_val_score = 0

            self.mixed_fp = self.cfg.MODEL.MIXED_PRECESION
            # gradient scaling for mixed-precision training
            self.scaler = GradScaler(
                backoff_factor=0.91, growth_factor=1.1, growth_interval=5000) \
                if self.mixed_fp else None             

        if self.cfg.AUGMENT.MIXUP.ENABLED:
            self.mixuper = Mixup(bs = self.cfg.SOLVER.SAMPLES_PER_BATCH, 
                beta=self.cfg.AUGMENT.MIXUP.BETA)

        if self.cfg.SOLVER.SWA.ENABLED and not self.cfg.SOLVER.TEST_ONLY:
            self.swa_start = self.cfg.SOLVER.SWA.START_ITER
            self.swa_model, self.swa_scheduler = build_swa_model(
                self.cfg, self.model, self.optimizer, self.device,
                is_pretrained=self.cfg.MODEL.PRE_TRAIN is not None)

    def train(self):
        self.model.train()
        if self.tail_only_iter > 0:
            self.freeze_tail()

        timer = utility.timer()
        for i in range(self.iter_start, self.iteration_total):
            if self.tail_only_iter > 0 and i > self.tail_only_iter:
                self.freeze_tail(defrost=True)
                self.tail_only_iter = -1 # only defrost once

            lr, hr, _ = next(self.loader_train)
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            if hasattr(self, 'mixuper'):
                lr, hr = self.mixuper(lr, hr)

            # full-precision when finetuning tail
            autocast_enabled = self.mixed_fp and self.tail_only_iter <= 0
            with autocast(enabled=autocast_enabled):
                sr = self.model(lr)
                loss = self.loss(sr, hr)

            self.optimizer.zero_grad(set_to_none = not self.mixed_fp)
            if autocast_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.scheduler_step(i) # update SWA model if it exists
            if (self.rank is None or self.rank == 0) and i % 10 == 0:
                self.log_train(i, loss, timer)

            del lr, hr, sr, loss # Release some GPU memory

            if (self.rank is None or self.rank == 0): # run inference in rank 0 process
                is_swa = hasattr(self, 'swa_model') and (i+1) > self.swa_start
                if (i+1) % self.cfg.SOLVER.TEST_EVERY == 0:
                    self.test(i+1, is_swa=is_swa)
                    self.model.train()

                if (i+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                    self.ckp.save(self, i+1, False, self.iter_start, is_swa,
                                  iter_suffix = True)

        # save stochastic weight averaging model
        self.maybe_save_swa_model()

    def log_train(self, i, loss, timer):
        lr = self.optimizer.param_groups[0]['lr']
        total_time = timer.toc()
        avg_itertime = total_time / (i+1-self.iter_start)
        est_timeleft = avg_itertime * (self.iteration_total - i) / 3600
        print(
            "[Iteration %05d] Loss: %.5f, LR: %.5f, " % (i+1, loss.item(), lr)
            + "Iter time: %.4fs, Total time: %.2fh, Time Left %.2fh." % (
                avg_itertime, total_time / 3600, est_timeleft))

        if i % 500 == 0 and i > 0 and torch.cuda.is_available():
            GPUtil.showUtilization(all=True)
            if self.mixed_fp:
                print(self.scaler.state_dict())

    def freeze_tail(self, defrost: bool = False):
        if defrost:
            print("Defrost all frozen layers.")
            for param in self.model.parameters():
                param.requires_grad = True
            return

        # keep only the tail trainable
        for name, param in self.model.named_parameters():
            prefix = ".".join(name.split(".")[:3])
            if prefix != "module.model.tail":
                param.requires_grad = False
            elif self.rank == 0:
                print("{} is trainable.".format(name))

    def test(self, iteration: int = -1, is_swa: bool = False):
        if is_swa:
            self.swa_model.eval()
            log_str = '\nSWA Evaluation:'
        else:
            self.model.eval()
            log_str = '\nEvaluation:'

        torch.set_grad_enabled(False)
        if self.ckp is not None:
            self.ckp.write_log(log_str)
            self.ckp.add_log(
                torch.zeros(1, len(self.loader_test), len(self.scale)))

            timer_test = utility.timer()
            if self.cfg.LOG.SAVE_RESULTS:
                self.ckp.begin_background()
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr = lr.to(self.device, non_blocking=True)
                        with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                            sr = self.swa_model(lr) if is_swa else self.model(lr)
                        sr = utility.quantize(sr, self.cfg.DATASET.RGB_RANGE)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr_torch(
                            sr, hr.to(sr.device), scale, float(self.cfg.DATASET.RGB_RANGE))

                        if self.cfg.LOG.SAVE_GT:
                            save_list.extend([lr, hr])

                        if self.cfg.LOG.SAVE_RESULTS:
                            self.ckp.save_results(
                                d, filename[0], save_list, scale)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} @iter {} (Best: {:.3f} @iter {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            iteration,
                            best[0][idx_data, idx_scale],
                            (best[1][idx_data, idx_scale] + 1) *
                            self.cfg.SOLVER.TEST_EVERY + self.iter_start
                        )
                    )

            if iteration >= 0:
                self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
                self.ckp.write_log('Saving...')

                if self.cfg.LOG.SAVE_RESULTS:
                    self.ckp.end_background()

                if not self.cfg.SOLVER.TEST_ONLY:
                    is_best = False
                    if best[0][idx_data, idx_scale] > self.best_val_score:
                        self.best_val_score = best[0][idx_data, idx_scale]
                        is_best = True
                    self.ckp.save(self, iteration, is_best, self.iter_start,
                                  is_swa = is_swa)

                self.ckp.write_log(
                    'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

        torch.set_grad_enabled(True)

    def scheduler_step(self, iter_total: int):
        # When SWA model exists, update SWA weights then
        # execute the lr scheduler.
        if not hasattr(self, 'swa_model'):
            self.lr_scheduler.step()
            return

        swa_start = self.cfg.SOLVER.SWA.START_ITER
        swa_merge = self.cfg.SOLVER.SWA.MERGE_ITER
        if iter_total >= swa_start:
            if iter_total % swa_merge == 0:
                self.swa_model.update_parameters(self.model)
                if self.rank is None or self.rank == 0:
                    print("Just updated SWA model.")
            self.swa_scheduler.step()
        else:
            self.lr_scheduler.step()

    def maybe_save_swa_model(self):
        if not hasattr(self, 'swa_model'):
            return

        # save swa model
        if self.rank is None or self.rank == 0:
            print("Save SWA model checkpoint.")
            filename = os.path.join(
                self.ckp.get_path('model'), 'model_swa.pth.tar')
            state = {'state_dict': self.swa_model.module.module.model.state_dict()}
            torch.save(state, filename)

        # maybe run test with swa model?
        self.ckp.write_log('Writing SWA Results')
        self.model.module.model.load_state_dict(self.swa_model.module.module.model.state_dict())
        self.test(-1)
