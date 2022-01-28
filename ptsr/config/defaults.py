from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.DEBUG = False
# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.DISTRIBUTED = False
_C.SYSTEM.PARALLEL = 'DP'
_C.SYSTEM.NUM_CPU = 4
_C.SYSTEM.NUM_GPU = 1

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Mixed-precision training/inference
_C.MODEL.MIXED_PRECESION = False

_C.MODEL.ACT_MODE = 'relu'
_C.MODEL.PRE_TRAIN = None
_C.MODEL.CKP_STRICT = True
_C.MODEL.CKP_IGNORE = None

_C.MODEL.EXTEND = '.'
_C.MODEL.BLOCK_TYPE = 'basicblock'
_C.MODEL.N_RESBLOCKS = 8
_C.MODEL.N_RESGROUPS = 8
_C.MODEL.OUT_CONV = False  # additional conv for each residual group
_C.MODEL.PLANES = 64
_C.MODEL.STOCHASTIC_DEPTH = False
_C.MODEL.STOCHASTIC_DEPTH_PROB = None # survival probability
_C.MODEL.ZERO_INIT_RESIDUAL = True
_C.MODEL.MULT_FLAG = True
_C.MODEL.EXPANSION = 1
_C.MODEL.KERNEL_SIZE = 3
_C.MODEL.SHIFT_MEAN = True
_C.MODEL.DILATION = False
_C.MODEL.SE_REDUCTION = 4
_C.MODEL.SHORT_SKIP = True
_C.MODEL.AFFINE_INIT_W = 0.1
_C.MODEL.DEFORM_CONV = False

# Standard deviation for normal distribution initialization
_C.MODEL.NORMAL_INIT_STD = None

_C.MODEL.RES_SCALE = 0.1
_C.MODEL.RES_SCALE_LEARNABLE = False

_C.MODEL.ENSEMBLE = CN({"ENABLED":False})
_C.MODEL.ENSEMBLE.MODE = 'mean' # mean, median

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATA_DIR = '/datasets/SR/BIX2X3X4'
_C.DATASET.DEMO_DIR = '../test'
_C.DATASET.DATA_TRAIN = ['DF2K']
_C.DATASET.DATA_VAL = ['DF2K']
_C.DATASET.DATA_TEST = ['DF2K', 'Set5',
                        'Set14C', 'B100', 'Urban100', 'Manga109']
_C.DATASET.DATA_RANGE = [[1, 3550], [3551, 3555]]
_C.DATASET.DATA_EXT = 'bin'  # 'bin', 'sep' or 'img'
_C.DATASET.DATA_SCALE = [4]
_C.DATASET.OUT_PATCH_SIZE = 192  # square patch size of model output
_C.DATASET.RGB_RANGE = 255
_C.DATASET.CHANNELS = 3

_C.DATASET.CHOP = False
_C.DATASET.CHOP_PAD = 20
_C.DATASET.CHOP_THRES = 160000

_C.DATASET.OVERLAP = CN({"ENABLED": False})
_C.DATASET.OVERLAP.STRIDE = 48
_C.DATASET.OVERLAP.SIZE = 96

_C.DATASET.REJECTION_SAMPLING = CN({"ENABLED": False})
_C.DATASET.REJECTION_SAMPLING.MAX_PSNR = 25.0
_C.DATASET.REJECTION_SAMPLING.PROB = 0.2

_C.DATASET.FINETUNE = CN({"ENABLED": False})
_C.DATASET.FINETUNE.DATA = ['Set5']


# -----------------------------------------------------------------------------
# Augment
# -----------------------------------------------------------------------------
_C.AUGMENT = CN({"ENABLED": True})
_C.AUGMENT.CHANNEL_SHUFFLE = False
_C.AUGMENT.INVERT = False
_C.AUGMENT.MIXUP = CN({"ENABLED": False})
_C.AUGMENT.MIXUP.BETA = 0.15

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

# Solver type needs to be one of 'SGD', 'Adam', 'AdamW'
_C.SOLVER.NAME = 'Adam'

_C.SOLVER.EPS = 1e-6 # change to 1e-4 in mixed-precision training
# Set the trust ratio clamp value to a large number (e.g., 1e6) to
# avoid truncating trust ratio in LAMB16 optimizer.
_C.SOLVER.CLAMP_TRUST_RATIO = 1e6

# Specify the learning rate scheduler.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

# Save a checkpoint after every this number of iterations.
_C.SOLVER.ITERATION_SAVE = 20000
_C.SOLVER.ITERATION_TOTAL = 200000
_C.SOLVER.TEST_EVERY = 1000

# Whether or not to restart training from iteration 0 regardless
# of the 'iteration' key in the checkpoint file. This option only
# works when a pretrained checkpoint is loaded (default: False).
_C.SOLVER.ITERATION_RESTART = False

_C.SOLVER.TAIL_ONLY_ITER = -1

_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.MIN_LR = 0.0

_C.SOLVER.MOMENTUM = 0.9  # SGD
_C.SOLVER.BETAS = (0.9, 0.999)  # ADAM

# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

# The iteration number to decrease learning rate by GAMMA
_C.SOLVER.GAMMA = 0.5

# should be a tuple like (30000,)
_C.SOLVER.STEPS = (300000, )

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000

_C.SOLVER.WARMUP_ITERS = 0

_C.SOLVER.WARMUP_METHOD = "linear"

# Number of samples per GPU. If we have 8 GPUs and SAMPLES_PER_BATCH = 8,
# then each GPU will see 8 samples and the effective batch size is 64.
_C.SOLVER.SAMPLES_PER_BATCH = 8

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Stochastic Weight Averaging
_C.SOLVER.SWA = CN({"ENABLED": False})
_C.SOLVER.SWA.LR_FACTOR = 0.1
_C.SOLVER.SWA.START_ITER = 150000
_C.SOLVER.SWA.MERGE_ITER = 50

_C.SOLVER.TEST_ONLY = False
_C.SOLVER.GAN_K = 1

# Learning rate factor for the residual blocks. It can be used with
# stochastic depth to balance the skipped blocks.
_C.SOLVER.RESIDUAL_LR_FACTOR = None

_C.SOLVER.LOSS = CN()
_C.SOLVER.LOSS.CONFIG = '1*L1'
_C.SOLVER.LOSS.SKIP_THRES = 1e8
# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.SAVE = 'test'
_C.LOG.LOAD = ''
_C.LOG.RESUME = 0
_C.LOG.SAVE_MODELS = False
_C.LOG.PRINT_EVERY = 100
_C.LOG.SAVE_RESULTS = True
_C.LOG.SAVE_GT = True


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
