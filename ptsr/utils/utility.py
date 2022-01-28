from typing import Optional
import torch.optim.lr_scheduler as lrs
import torch.optim as optim
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log10, sqrt
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart:
            self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.datatest = []

        if (self.cfg.SOLVER.TEST_EVERY and not self.cfg.SOLVER.TEST_ONLY):
            self.datatest = self.cfg.DATASET.DATA_VAL
        elif (self.cfg.SOLVER.TEST_ONLY):
            self.datatest = self.cfg.DATASET.DATA_TEST

        time_stamp = datetime.datetime.now().strftime("_%b%d%y_%H%M")
        if not cfg.LOG.LOAD:
            if not cfg.LOG.SAVE:
                cfg.LOG.SAVE = now
            self.dir = os.path.join('outputs', cfg.LOG.SAVE + time_stamp)
        else:
            self.dir = os.path.join('outputs', cfg.LOG.LOAD + time_stamp)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(
                    len(self.log)*self.cfg.SOLVER.TEST_EVERY))
            else:
                cfg.LOG.LOAD = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in self.datatest:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.yaml'), open_type) as f:
            print(cfg, file=f)

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, iteration, is_best=False, iter_start=0, is_swa=False, iter_suffix=False):
        self.save_model(self.get_path('model'), trainer, iteration, is_best, is_swa, iter_suffix)
        self.plot_psnr(iteration, iter_start)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def save_model(self, apath, trainer, iteration: int, is_best: bool = False,
                   is_swa: bool = False, iter_suffix: bool = False):
        save_dirs = [os.path.join(apath, 'model_latest.pth.tar')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pth.tar'))
        elif iter_suffix:
            save_dirs.append(os.path.join(apath, 'model_%06d.pth.tar' % iteration))

        if is_swa:
            state_dict = trainer.swa_model.module.module.model.state_dict()
        else:
            state_dict = trainer.model.module.model.state_dict()  # DP, DDP
        state = {'iteration': iteration,
                 'state_dict': state_dict,
                 'optimizer': trainer.optimizer.state_dict(),
                 'lr_scheduler': trainer.lr_scheduler.state_dict()}

        if hasattr(trainer, 'mixed_fp') and trainer.mixed_fp:
            state['scaler'] = trainer.scaler.state_dict()

        for filename in save_dirs:
            torch.save(state, filename)

    def load_model(self, pre_train, trainer, device, restart: bool = False,
                   test_mode: bool = False, strict: bool = True, 
                   ignore: Optional[str] = None):
        if pre_train is None:
            return

        state = torch.load(pre_train, map_location=device)
        if isinstance(state['state_dict'], tuple):
            trainer.model.module.model.load_state_dict(
                state['state_dict'][0], strict=strict)            
        else:
            pretrained_dict = state['state_dict']
            if ignore is not None:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if ignore not in k}
            trainer.model.module.model.load_state_dict(pretrained_dict, strict=strict)

        if not restart and not test_mode:
            trainer.optimizer.load_state_dict(state['optimizer'])
            trainer.lr_scheduler.load_state_dict(state['lr_scheduler'])
            trainer.iter_start = state['iteration']
            if hasattr(trainer, 'mixed_fp') and trainer.mixed_fp and 'scaler' in state:
                trainer.scaler.load_state_dict(state['scaler'])

        del state # release GPU memory 

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def plot_psnr(self, iteration, iter_start=0):
        intervel = self.cfg.SOLVER.TEST_EVERY
        num_points = (iteration + 1 - iter_start) // intervel
        axis = list(range(1, num_points+1))
        axis = np.array(axis) * intervel + iter_start
        for idx_data, d in enumerate(self.datatest):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.cfg.DATASET.DATA_SCALE):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Iterations')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None:
                        break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]

        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.cfg.LOG.SAVE_RESULTS:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.cfg.DATASET.RGB_RANGE)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

    def done(self):
        self.log_file.close()


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr_torch(sr, hr, scale, rgb_range: float = 255.0,
                    use_gray_coeffs: bool = True):
    # Input images should be in (B, C, H, W) format.
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    diff = diff[..., scale:-scale, scale:-scale]
    if diff.size(1) > 1 and use_gray_coeffs:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / rgb_range
        diff = diff.mul(convert).sum(dim=1)

    mse = diff.pow(2).mean()
    if mse == 0:  # PSNR have no importance.
        return 100

    return -10 * log10(mse)


def calc_psnr_numpy(sr, hr, scale, rgb_range: float = 255.0):
    # Input images should be in (H, W, C) format.
    diff = (sr - hr)
    diff = diff[scale:-scale, scale:-scale]

    mse = np.mean(diff ** 2)
    if mse == 0:  # PSNR have no importance.
        return 100

    psnr = 20 * log10(rgb_range / sqrt(mse))
    return psnr
