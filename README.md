# AttaNet: Attention-Augmented Network for Fast and Accurate Scene Parsing(AAAI21)
## Introduction
In this paper, we propose a new model, called Attention-Augmented Network (AttaNet), to capture both global context and multi-level semantics while keeping the efficiency high. Not only did our network achieve the leading performance on Cityscapes and ADE20K, SAM and AFM can also be combined with different backbone networks to achieve different levels of speed/accuracy trade-offs. Specifically, our approach obtains 79.9%, 78.5%, and 70.1% mIoU scores on the Cityscapes test set while keeping a real-time speed of 71 FPS, 130 FPS, and 180 FPS respectively on GTX 1080Ti. ![results](https://github.com/songqi-github/AttaNet/blob/main/figs/results.png)
Please refer to our paper for more details:
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16359), [arxiv version](https://arxiv.org/abs/2103.05930)

## Segmentation Models:
Please download the trained model, the mIoU is evaluate on Cityscape validation dataset.

|      Model       | Train Set | Test Set | mIoU (%) |                             Link                             |
| :--------------: | :-------: | :------: | :------: | :----------------------------------------------------------: |
|  AttaNet_light   |   Train   |   Val    |   70.6   | [BaiduYun(Access Code:zmb3)](https://pan.baidu.com/s/1OR45RYDU6sQ-jiIliisboA) |
| AttaNet_ResNet18 |   Train   |   Val    |   78.8   | [BaiduYun(Access Code:66tx)](https://pan.baidu.com/s/1L1nyyz1F48aI4ODYX9euUg) |

## Quick start
Download pretrained models for resnet series.
```
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
```
#### Training
The training settings require GPU with at least 11GB memory.
```
python -m torch.distributed.launch --nproc_per_node=2 train.py
```
#### Evaluating
Evaluating AttaNet on the Cityscape validation dataset.
```
python evaluate.py  # for accuracy testing of heavy models
python realtime_evaluate.py  # for accuracy testing of real-time models
```

## Citation
If you find this repo is useful for your research, Please consider citing our paper:

```
@inproceedings{song2021attanet,
  title={AttaNet: Attention-Augmented Network for Fast and Accurate Scene Parsing},
  author={Song, Qi and Mei, Kangfu and Huang, Rui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2567--2575},
  year={2021}
}

@article{Song2021AttaNetAN,
  title={AttaNet: Attention-Augmented Network for Fast and Accurate Scene Parsing},
  author={Qi Song and Kangfu Mei and Rui Huang},
  journal={ArXiv},
  year={2021},
  volume={abs/2103.05930}
}
```
