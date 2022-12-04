# Fast ImageNet training for 100 epochs with ResNet-50
Code is borrowed from ([link](https://github.com/snu-mllab/Co-Mixup)). We just insert our method to this codebase.
Here, **we use Distributed Data Parallel (DDP)** rather than Data Parallel (DP).

## Requirements
* Python 3.7
* PyTorch 1.7.1
* [Apex](https://github.com/NVIDIA/apex) (to use half precision speedup). 

Apex can only be run with the right CUDA version in the machine and the version that PyTorch is installed with.
## Preparing ImageNet Data
1. Download and prepare the ImageNet dataset. You can use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh), 
provided by the PyTorch repository, to move the validation subset to the labeled subfolders.
2. Prepare resized versions of the ImageNet dataset by running

```
python resize.py --path path/to/imagenet --dest /path/to/extractDestination
```

## Reproducing the results
To reproduce the results from the paper, specifiy the path in the `sh` script made with `resize.py`:
```
sh run_fast_ddp.sh /path/to/DATA160 /path/to/DATA352 rlmix
```
This script runs the main code `main_fast_ddp.py` using the configurations provided in the `configs` folder. 
All parameters can be modified by adjusting the configuration files in the `configs/` folder.

