import os
from ptsr.data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, cfg, name='DIV2K', train=True, benchmark=False):
        data_range = cfg.DATASET.DATA_RANGE
        if train:
            data_range = data_range[0]
        else:
            if cfg.SOLVER.TEST_ONLY and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = data_range
        super().__init__(
            cfg, name=name, train=train, benchmark=benchmark
        )

    # def _scan(self):
    #     names_hr, names_lr = super(DIV2K, self)._scan()
    #     names_hr = names_hr[self.begin - 1:self.end]
    #     names_lr = [n[self.begin - 1:self.end] for n in names_lr]

    #     return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

