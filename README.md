# AttaNet: Attention-Augmented Network for Fast and Accurate Scene Parsing(AAAI21)
## Introduction
In this paper, we propose a new model, called Attention-Augmented Network (AttaNet), to capture both global context and multi-level semantics while keeping the efficiency high. Not only did our network achieve the leading performance on Cityscapes and ADE20K, SAM and AFM can also be combined with different backbone networks to achieve different levels of speed/accuracy trade-offs. Specifically, our approach obtains 79.9%, 78.5%, and 70.1% mIoU scores on the Cityscapes test set while keeping a real-time speed of 71 FPS, 130 FPS, and 180 FPS respectively on GTX 1080Ti.

Please refer to our paper for more details:
[arxiv version](https://arxiv.org/abs/2103.05930)

## Code
AttaNet head is uploaded, but it is not the final version. You can check how to implement SAM and AFM. I'm still working on this repo.

## Segmentation Models:
Please download the trained model, the mIoU is evaluate on Cityscape validation dataset.

|      Model       | Train Set | Test Set | mIoU (%) | Link |
| :--------------: | :-------: | :------: | :------: | :--: |
|  AttaNet_light   |   Train   |   Val    |          |      |
| AttaNet_ResNet18 |   Train   |   Val    |          |      |


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
### Training
```
python -m torch.distributed.launch --nproc_per_node=2 train.py
```
### Evaluating
Evaluating AttaNet on the Cityscape validation dataset.
```
python evaluate.py
```

## Citation
If you find this repo is useful for your research, Please consider citing our paper:

```
@article{Song2021AttaNetAN,
  title={AttaNet: Attention-Augmented Network for Fast and Accurate Scene Parsing},
  author={Qi Song and Kangfu Mei and Rui Huang},
  journal={ArXiv},
  year={2021},
  volume={abs/2103.05930}
}
```
