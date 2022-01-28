# Revisiting RCAN: Improved Training for Image Super-Resolution

## Introduction

Image super-resolution (SR) is a fast-moving field with novel architectures attracting the spotlight. However, most SR models were optimized with dated training strategies. In this work, we revisit the popular RCAN model and examine the effect of different training options in SR. Surprisingly (or perhaps as expected), we show that RCAN can outperform or match nearly all the CNN-based SR architectures published after RCAN on standard benchmarks with a proper training strategy and minimal architecture change. Besides, although RCAN is a very large SR architecture with more than four hundred convolutional layers, we draw a notable conclusion that underfitting is still the main problem restricting the model capability instead of overfitting. We observe supportive evidence that increasing training iterations clearly improves the model performance while applying regularization techniques generally degrades the predictions. We denote our simply revised RCAN as **RCAN-it** and recommend practitioners to use it as baselines for future research. Please check our [**pre-print**](https://arxiv.org/abs/2201.11279) for more information.

## Environment Setup

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

## Multi-processing Distributed Data Parallel Training

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

## Data Parallel Training

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

## Citation

Please check this [pre-print](https://arxiv.org/abs/2201.11279) for details. If you find this work useful for your research, please cite:

```bibtex
@misc{lin2022revisiting,
      title={Revisiting RCAN: Improved Training for Image Super-Resolution}, 
      author={Zudi Lin and Prateek Garg and Atmadeep Banerjee and Salma Abdel Magid and Deqing Sun and Yulun Zhang and Luc Van Gool and Donglai Wei and Hanspeter Pfister},
      year={2022},
      eprint={2201.11279},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
