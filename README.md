# Scalable Penalized Regression for Noise Detection in Learning With Noisy Labels
\[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Scalable_Penalized_Regression_for_Noise_Detection_in_Learning_With_Noisy_CVPR_2022_paper.pdf)\]
\[[intro](https://yikai-wang.github.io/spr/)\]

## Overview
This is the official repo for our CVPR22 paper: Scalable Penalized Regression for Noise Detection in Learning With Noisy Labels.

> SPR is a theoretically guaranteed noisy label detection framework to detect and remove noisy data for learning with noisy labels. It produces a penalized regression to model the linear relation between network features and one-hot labels, where the noisy data are identified by the non-zero mean shift parameters solved in the regression model. A non-asymptotic probabilistic condition for SPR is provided to correctly identify the noisy data. SPR can also be combined with semi-supervised algorithm to further exploit the support of noisy data as unlabeled data.

## Requirements
```
python==3.7.6
numpy==1.19.1
scipy==1.6.0
scikit-learn==0.23.2
torch==1.5.1
torchvision==0.6.0a0+35d732a
```

## Data Preparing

MNIST and CIFAR-10 can be downloaded using *torchvision*. The other two datasets can be downloaded from the official link: [ANIMAL10](https://dm.kaist.ac.kr/datasets/animal-10n/), [WebVision](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html).

The datasets are expected to be stored in the folder **../data** or specified by the *root* parameter, and arranged as follows:
```
│data/
├── MNIST/
│   ├── ......
├── CIFAR10/
│   ├── ......
├── animal10/
│   ├── training/
│   │   ├── ......
│   ├── testing/
│   │   ├── ......
├── webvision/
│   ├── info/
│   │   ├── ......
│   ├── google/
│   │   ├── ......
│   ├── val_images_256/
│   │   ├── ......
(Optional)
├── imagenet/
│   ├── meta.mat
│   ├── ILSVRC2012_validation_ground_truth.txt
│   ├── val/
│   │   ├── ......
```


## Pretrained Model
The pretained models can be downloaded from [here](https://drive.google.com/drive/folders/1m0SDABpEcJotp1bnbYILP2KnAf2XGPwX?usp=sharing) and should be put in the folder **ckpt**.

## Training
Example training commands are listed in the folder **scripts**.
You could try the following commands as a start.

**Note**: To train with SPR but without using CutMix, you should set ```--cutmix 1``` and ```--cutmix_prob 0```.

Train SPR on MNIST with different noise setting:
```
python scripts/train_mnist.py
```

Train SPR on CIFAR10 with different noise setting:
```
python scripts/train_cifar.py
```

Train SPR on Animal10:
```
python scripts/train_animal.py
```

Train SPR on WebVision:
```
python scripts/train_webvision.py
```

## Evaluation
Example evaluation commands are listed in the folder **scripts**.
You could try the following commands as a start.

Test SPR on MNIST with different noise setting:
```
python scripts/eval_mnist.py
```

Test SPR on CIFAR10 with different noise setting:
```
python scripts/eval_cifar.py
```

Test SPR on Animal10:
```
python scripts/eval_animal.py
```

Test SPR on WebVision:
```
python scripts/eval_webvision.py
```

## Acknowledgements
Thanks to everyone who makes their code and models available. In particular,

- [CutMix](https://github.com/clovaai/CutMix-PyTorch)
- [SR](https://github.com/hitcszx/lnl_sr)
- [TopoFilter](https://github.com/pxiangwu/TopoFilter)

## Contact Information
For issues using SPR, please submit a GitHub issue.

## Citation

If you found the provided code useful, please cite our work.

```
@inproceedings{wang2022scalable,
  title={Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels},
  author={Wang, Yikai and Sun, Xinwei and Fu, Yanwei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```