# Revisiting RCAN: Improved Training for Image Super-Resolution

## Table of contents
1. [Introduction](#intro)
2. [Pre-trained Weights](#pretrained)
3. [Installation](#install)
4. [Distributed Training](#ddp)
5. [Data-paralell Training](#dp)
6. [Custom Dataset](#custom_data)
7. [Citation](#citation)

## Introduction <a name="intro"></a>

Image super-resolution (SR) is a fast-moving field with novel architectures attracting the spotlight. However, most SR models were optimized with dated training strategies. In this work, we revisit the popular RCAN model and examine the effect of different training options in SR. Surprisingly (or perhaps as expected), we show that RCAN can outperform or match nearly all the CNN-based SR architectures published after RCAN on standard benchmarks with a proper training strategy and minimal architecture change. Besides, although RCAN is a very large SR architecture with more than four hundred convolutional layers, we draw a notable conclusion that underfitting is still the main problem restricting the model capability instead of overfitting. We observe supportive evidence that increasing training iterations clearly improves the model performance while applying regularization techniques generally degrades the predictions. We denote our simply revised RCAN as **RCAN-it** and recommend practitioners to use it as baselines for future research. Please check our [**pre-print**](https://arxiv.org/abs/2201.11279) for more information.

## Pre-trained Weights <a name="pretrained"></a>

We release the pre-trained RCAN-it weights for different scales:

[[RCAN-it (x2)](https://drive.google.com/uc?export=download&id=1g7ch--BAgxc8L4p4ERoth-f_NbyPaWIt)] [[RCAN-it (x3)](https://drive.google.com/uc?export=download&id=1l0q0RfKMyfya8mDKDCmsIvg7XAoF40S-)] [[RCAN-it (x4)](https://drive.google.com/uc?export=download&id=1dDxpjTKtCILBONcEkcI8YAdcG3Nswgvk)]

We also share the [predictions](https://drive.google.com/uc?export=download&id=1aRGAttp2G4qY7WvcCXg2UbGUyklvWhXm) on the Set5 benchmark dataset. The scores (PSNR and SSIM) are evaluated using the MATLAB code in the [RCAN](https://github.com/yulunzhang/RCAN/blob/master/RCAN_TestCode/Evaluate_PSNR_SSIM.m) repository.

## Installation <a name="install"></a>

Create a new conda environment and install PyTorch:

```shell
conda create -n ptsr python=3.8 numpy
conda activate ptsr
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
```

Install the required packages:

```shell
git clone https://github.com/zudi-lin/rcan-it.git
cd rcan-it
pip install --editable .
```

Our package is called **ptsr**, abbreviating *A PyTorch Framework for Image Super-Resolution*. Then run tests to validate the installation:

```shell
python -m unittest discover -b tests
```

## Distributed Data Parallel Training (Recommended) <a name="ddp"></a>

For different hardware conditions, please first update the config files accordingly. Even for single-node single-GPU training, we use distributed data parallel (DDP) for consistency.

### Single Node

Single GPU training:

```shell
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run --nproc_per_node=1 \
--master_port=9988 main.py --distributed --config-base configs/RCAN/RCAN_Improved.yaml \
--config-file configs/RCAN/RCAN_x2.yaml
```

Single node with multiple (*e.g.*, 4) GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nproc_per_node=4 \
--master_port=9977 main.py --distributed --config-base configs/RCAN/RCAN_Improved.yaml \
--config-file configs/RCAN/RCAN_x2.yaml
```

By default the configuration file, model checkpoints and validation curve will be saved
under `outputs/`, which is added to `.gitignore` and will be untracked by Git.

### Multiple Nodes

After activating the virtual environment with PyTorch>=1.9.0, run `hostname -I | awk '{print $1}'` to get the ip address of the master node. Suppose the master ip address is `10.31.133.85`, and we want to train the model on two nodes with multiple GPUs, then the commands are:

*Node 0* (master node):

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 \ 
--node_rank=0 --master_addr="10.31.133.85" --master_port=9922 main.py --distributed \
--config-base configs/RCAN/RCAN_Improved.yaml --config-file configs/RCAN/RCAN_x2.yaml
```

*Node 1*:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 \ 
--node_rank=1 --master_addr="10.31.133.85" --master_port=9922 main.py --distributed \
--config-base configs/RCAN/RCAN_Improved.yaml --config-file configs/RCAN/RCAN_x2.yaml
```

Description of the options:

- `--nproc_per_node`: number of processes on each node. Set this to the number of GPUs on the node to maximize the training efficiency.
- `--nnodes`: total number of nodes for training.
- `--node_rank`: rank of the current node within all nodes.
- `--master_addr`: the ip address of the master (rank 0) node.
- `--master_port`: a free port to communicate with the master node.
- `--distributed`: multi-processing Distributed Data Parallel (DDP) training.
- `--local_world_size`: number of GPUs on the current node.

For a system with Slurm Workload Manager, please load required modules: `module load cuda cudnn`.

### Inference

We recommend to continue using DDP during inference for consistency. The exemplary inference command using one GPU device is:

```shell
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.run --nproc_per_node=1 \
--master_port=9992 main.py --distributed --config-base configs/RCAN/RCAN_Improved.yaml \
--config-file configs/RCAN/RCAN_x2.yaml MODEL.PRE_TRAIN RCAN_pretrained/model_best.pth.tar \
SOLVER.TEST_ONLY True MODEL.ENSEMBLE.ENABLED True DATASET.CHOP True
```

Please remember to update `SYSTEM.NUM_GPU` and `SYSTEM.NUM_CPU` based on your system specifications.

## Data Parallel Training <a name="dp"></a>

Data Parallel training only works on single node with one or multiple GPUs. Different from
the DDP scheme, it will create only one process. Single GPU training:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --config-base configs/RCAN/RCAN_Base.yaml \
--config-file configs/RCAN/RCAN_x2.yaml
```

Single node with multiple (e.g., 4) GPUs:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config-base configs/RCAN/RCAN_Base.yaml \
--config-file configs/RCAN/RCAN_x2.yaml
```

## Training on New Datasets <a name="custom_data"></a>

Our code by default train on the **DF2K** dataset. To train on your own dataset (supposing the dataset contains 99 training images in this example). First, structure the data as follows in the dataset directory:

```
MyData/
  MyData_train_HR/
    0001.png
    0002.png
    ...
    0099.png
  MyData_train_LR_bicubic/
    X2/
      0001x2.png
      ...
    X3/
      0001x3.png
      ...
    X4/
      0001x4.png
      ...
```

Then update the configuration options in the YAML file:

```yaml
DATASET:
  DATA_EXT: bin
  DATA_DIR: path/to/data
  DATA_TRAIN: ['MyData']
  DATA_VAL: ['MyData']
  DATA_RANGE: [[1, 95], [96, 99]] # split training and validation
```

Note that `DATASET.DATA_EXT: bin` will create a `bin` folder in the dataset directory and save individual images as a single binary file for fast data loading.


## Citation <a name="citation"></a>

Please check this [pre-print](https://arxiv.org/abs/2201.11279) for details. If you find this work useful for your research, please cite:

```bibtex
@article{lin2022revisiting,
  title={Revisiting RCAN: Improved Training for Image Super-Resolution},
  author={Lin, Zudi and Garg, Prateek and Banerjee, Atmadeep and Magid, Salma Abdel and Sun, Deqing and Zhang, Yulun and Van Gool, Luc and Wei, Donglai and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2201.11279},
  year={2022}
}
```
