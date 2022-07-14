# GradAuto
This repository contains the result and the sample code for the work:
GradAuto: Energy-oriented Attack on Dynamic Neural Networks. 

# To perturb adversarial samples to SkipNet on the ImageNet validation dataset
## Prerequisite 
1. We support training with Pytorch 1.10.0. To install required packages
```
conda install pytorch=1.10 torchvision cudatoolkit=<the CUDA version you want> numpy
```

2. To prepare ImageNet dataset, please follow this [link](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

## Training 
1. To train the adversarial samples with $K=1$, run
```
python -u train_autograd.py --model-type rl --K 1
```
2. To train the adversarial samples without accuracy drop, run
```
python -u train_autograd.py --model-type rl --K 1 --acc-maintain
```