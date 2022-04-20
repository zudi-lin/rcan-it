import os
from ptsr.data import srdata

class CustomData(srdata.SRData):
    def __init__(self, cfg, name='MyData', train=True, benchmark=False):
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

    def _set_filesystem(self, dir_data):
        super()._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, self.name + '_train_HR')
        self.dir_lr = os.path.join(self.apath, self.name + '_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
