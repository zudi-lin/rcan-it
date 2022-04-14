from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset


class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'):
                d.set_scale(idx_scale)


class Data(object):
    def __init__(self, cfg, rank):
        if (cfg.DATASET.FINETUNE.ENABLED):
            data_train = cfg.DATASET.FINETUNE.DATA
            print("Using Dataset(s): " + ','.join(data_train) + " for finetuning\n")
        else:
            data_train = cfg.DATASET.DATA_TRAIN
            print("Using Dataset(s): " + ','.join(data_train) + " for training\n")

        self.loader_train = None
        if not cfg.SOLVER.TEST_ONLY:
            datasets = []
            for d in data_train:
                if ("2K" in d):
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('ptsr.data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(cfg, rank=rank, name=d))

                if ("MyData" in d):
                    m = import_module('ptsr.data.custom')
                    datasets.append(getattr(m, "CustomData")(cfg, rank=rank, name=d))

                # finetune on the benchmark sets to get "oracle" performance
                if d in ['Set5', 'Set14C', 'B100', 'Urban100', 'Manga109']:
                    m = import_module('ptsr.data.benchmark')
                    datasets.append(getattr(m, 'Benchmark')(cfg, train=True, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=cfg.SOLVER.SAMPLES_PER_BATCH,
                shuffle=True,
                pin_memory=bool(cfg.SYSTEM.NUM_GPU),
                # distribute workers among mutiple processes
                num_workers=cfg.SYSTEM.NUM_CPU // cfg.SYSTEM.NUM_GPU)

        # build val and test loaders
        self.loader_test = []
        datatest = []

        if (cfg.SOLVER.TEST_EVERY and not cfg.SOLVER.TEST_ONLY):
            datatest = cfg.DATASET.DATA_VAL
        elif (cfg.SOLVER.TEST_ONLY):
            datatest = cfg.DATASET.DATA_TEST

        for d in datatest:
            if ("MyData" in d):
                m = import_module('ptsr.data.custom')
                testset = getattr(m, "CustomData")(cfg, train=False, name=d, rank=rank)
            elif d in ['Set5', 'Set14C', 'B100', 'Urban100', 'Manga109']:
                m = import_module('ptsr.data.benchmark')
                testset = getattr(m, 'Benchmark')(cfg, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('ptsr.data.' + module_name.lower())
                testset = getattr(m, module_name)(cfg, train=False, name=d, rank=rank)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=bool(cfg.SYSTEM.NUM_GPU),
                    num_workers=cfg.SYSTEM.NUM_CPU // cfg.SYSTEM.NUM_GPU))
