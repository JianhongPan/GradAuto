# GradAuto

This repository contains the result and the sample code for the work:
[GradAuto: Energy-oriented Attack on Dynamic Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3150_ECCV_2022_paper.php)
by
[Jianhong Pan](scholar.google.com/citations?user=J_IepqIAAAAJ), 
[Qichen Zheng](scholar.google.com/citations?user=d6AbpzgAAAAJ), 
[Zhipeng Fan](scholar.google.com/citations?user=Nb6ggPwAAAAJ), 
[Hossein Rahmani](scholar.google.com/citations?user=zFyT_gwAAAAJ),
[Qiuhong Ke](scholar.google.com/citations?user=84qxdhsAAAAJ), and 
[Jun Liu](scholar.google.com/citations?user=Q5Ild8UAAAAJ&hl)

### Citation

If you find our project useful in your research, please consider citing:

```
@inproceedings{pan2022gradauto,
  title={Gradauto: Energy-oriented attack on dynamic neural networks},
  author={Pan, Jianhong and Zheng, Qichen and Fan, Zhipeng and Rahmani, Hossein and Ke, Qiuhong and Liu, Jun},
  booktitle={European Conference on Computer Vision},
  pages={637--653},
  year={2022},
  organization={Springer}
}
```

# To perturb adversarial samples to SkipNet on the ImageNet validation dataset
## Prerequisite 
1. We support training with Pytorch 1.10.0. To install required packages
```
conda install pytorch=1.10 torchvision cudatoolkit=<the CUDA version you want> numpy
```

2. To prepare ImageNet dataset, please follow this [link](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

3. To prepare SkipNet pretrained model, please follow this [link](https://github.com/ucbdrive/skipnet/tree/master/imagenet).

## Training 
1. To train the adversarial samples with $K=1$, run
```
python -u train_autograd.py --model-type rl --K 1
```
2. To train the adversarial samples without accuracy drop, run
```
python -u train_autograd.py --model-type rl --K 1 --acc-maintain
```