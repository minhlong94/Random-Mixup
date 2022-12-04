# Official PyTorch implementation of R-Mix and RL-Mix

This is the official code release for the paper ["Expeditious Saliency-based Mixup through Random
Gradient Thresholding"](), accepted at [2nd International Workshop on Practical
Deep Learning in the Wild (Practical-DL)](https://practical-dl.github.io/#paper-submission) at AAAI Conference on Artificial Intelligence 2023.

![](.\assets\mixup vis final.png)
R-Mix aims to combine randomization through saliency-guided training of mix-up, which has shown to have a large improvement for CutMix.

R-Mix eliminates complex saliency optimization process by replacing it with a random procedure, which cuts down 1/3 of the training time.

## Requirements

In addition to dependencies listed in `requirements.txt` file, the code is tested with:

- Python 3.7+
- torch 1.7.1
- torchvision 0.8.2
- CUDA 11.0
- Ubuntu 18.04 (for ImageNet experiments)

The code is also tested with PyTorch 1.9. Simply install the requirements from the file:

```bash
# Install dependencies
pip install -r requirements.txt
```

If one wants to run ImageNet experiments, [Apex](https://github.com/NVIDIA/apex) is required. We use the codebase from 
[Co-Mixup](https://github.com/snu-mllab/Co-Mixup). With 4 RTX 2080Ti, it is expected to take about 5 days, and with 4
A6000, it takes about 1 day.

```bash
sh main_fast_ddp.sh /path/to/size160 /path/to/size352 rmix
```

## Checkpoints

We provide the pretrained models at this link: 
[Google Drive](https://drive.google.com/drive/folders/1TS4K2GTB_OTjMBPx3vBY8Jy_QEsmzCNJ?usp=sharing).
Last checkpoints showed:

| Model (Dataset)                      | Top-1 Accuracy |
|--------------------------------------|:--------------:|
| PreActResNet18 (CIFAR-100), CutMix+  |     81.62      |
| WideResNet28-10 (CIFAR-100), CutMix+ |     83.53      |
| PreActResNet18 (CIFAR-100), R-Mix    |     81.45      |
| WideResNet16-8 (CIFAR-100), R-Mix    |     82.32      |
| WideResNet28-10 (CIFAR-100), R-Mix   |     84.95      |
| ResNeXt29-4-24 (CIFAR-100), R-Mix    |     83.02      |
| ResNet-50 (ImageNet), R-Mix          |     77.39      |

We further improve CutMix+ to be even *better* than reported in the paper of PARN, but it is still fall short on bigger models. 

To validate the result, use the following preprocessing code (`requirements.txt` dependencies should be installed):

```python
import torch
from torchvision import transforms
from torchvision import datasets
import torchmetrics
from torchmetrics import MetricCollection

device = "cuda"
def calculate_metrics(model, dataloader):
    metrics = MetricCollection({
        "acc1": torchmetrics.Accuracy(top_k=1).to(device),
        "acc5": torchmetrics.Accuracy(top_k=5).to(device),
    })
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            metrics(torch.nn.Softmax(dim=1)(outputs), labels.to(device))
    print(f"{metrics.compute()}")


batch_size = 100
mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
cifar100_test = datasets.CIFAR100(root='./data', train=False,
                                 download=True, transform=test_transform)

test_loader = torch.utils.data.DataLoader(cifar100_test,
                                          batch_size=batch_size,
                                          shuffle=False)

from model.preactresnet import preactresnet18

model = preactresnet18(100, stride=1)  # Can use our code to initialize this model
checkpoint = torch.load('../<file>.pth.tar')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to(device)

calculate_metrics(model, test_loader)
```

For ImageNet, simply use `model = torch.load(...)` to load the `tar` file, and use `model.keys()` to get the detailed information.

Accuracies are different when evaluating on CPU and GPU. Please use GPU to validate the results.

## Reproduce results

Detailed descriptions of Argument Parser are provided in `parse_args.py` file. Hyperparameters are set to the best.
### CIFAR-100

To reproduce R-Mix with `OneCycleLR`, use:

```bash
python main.py
```

To reproduce RL-Mix with `OneCycleLR`, use:

```bash
python main.py --method rlmix
```

To reproduce the result with `MultiStepLR`, change the `scheduler` argument:

```bash
python main.py --scheduler MultiStepLR
```


R-Mix is expected to take about 4 hours, on one `NVIDIA Tesla P100-PCIE-16GB` and RL-Mix is about 8 hours. 
Currently, it only supports `stdout` log. For a more detailed logging, please use `wandb` as instructed below.

By default, we use 4 workers. One can change the number of workers in `main.py` for a fair comparison. However, on a Kaggle session it will
not make a big difference.

Final result may be different due to hardware/CUDA version/PyTorch version. However, on CIFAR-100, it should be at least 80% accuracy.

### ImageNet
Please see the folder `./imagenet`

## Wandb integration

We also implement `wandb` to monitor the training process. To use wandb, simply add 5 additional arguments:

```bash
python main.py ... --use_wandb 1 --wandb_key "XXX" --wandb_project "YYY" \
--wandb_entity "ZZZ" --wandb_name "YOUR_RUN_NAME"
```

Replace the key, project, entity and name to your own project.

## Visualization
Simply add `--vis_epoch 10` to visualize the mix-up images every 10 epochs. Default to 0, means no visualization.

## Single-file R-Mix implementation

We also provide a ready-to-use (user can directly copy it to your project) R-Mix. Code is in the
file `r_mix.py`.

## Citing
```
@inproceedings{luu2022rmix,
      title={Expeditious Saliency-based Mixup through Random Gradient Thresholding}, 
      author={Minh-Long Luu and Zeyi Huang and Eric P. Xing and Yong Jae Lee and Haohan Wang},
      booktitle={2nd Pracitcal-DL Workshop at AAAI},
      year={2023},
}
```

## License
MIT license.