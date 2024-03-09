# Distinguishing Homophenes using Multi-head Visual-audio Memory for Lip Reading
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/distinguishing-homophenes-using-multi-head-1/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=distinguishing-homophenes-using-multi-head-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/distinguishing-homophenes-using-multi-head-1/lipreading-on-lrw-1000)](https://paperswithcode.com/sota/lipreading-on-lrw-1000?p=distinguishing-homophenes-using-multi-head-1)

This repository contains the PyTorch implementation of the following paper:
> **Distinguishing Homophenes using Multi-head Visual-audio Memory for Lip Reading**<br>
> Minsu Kim, Jeong Hun Yeo, and Yong Man Ro<br>
> \[[Paper](https://www.aaai.org/AAAI22Papers/AAAI-6712.KimM.pdf)\]

<div align="center"><img width="75%" src="img/img.png?raw=true" /></div>

## Preparation

### Requirements
- python 3.7
- pytorch 1.6 ~ 1.8
- torchvision
- torchaudio
- ffmpeg
- av
- tensorboard
- scikit-image
- pillow

### Datasets
LRW dataset can be downloaded from the below link.
- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

The pre-processing will be done in the data loader.<br>
The video is cropped with the bounding box \[x1:59, y1:95, x2:195, y2:231\].

## Testing the Model
To test the model, run following command:
```shell
# Testing example for LRW
python test.py \
--lrw 'enter_data_path' \
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 80 \
--radius 16 --n_slot 112 --head 8 \
--test_aug \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint`: the checkpoint file
- `--batch_size`: batch size
- `--test_aug`: whether performing test time augmentation  `--distributed`: Use DataDistributedParallel  `--dataparallel`: Use DataParallel
- `--gpu`: gpu for using `--lr`: learning rate `--n_slot`: memory slot size `--radius`: scaling factor for addressing score `--head`: number of heads for visual-audio memory
- Refer to `test.py` for the other testing parameters

## Pretrained Models
You can download the pretrained models. <br>
Put the ckpt in './data/'

**Pretrained model**
- https://drive.google.com/file/d/10hhiYlhgHeW4-DtZdwSKZVcYuvKBkISs/view?usp=sharing

To test the pretrained model, run following command:
```shell
# Testing example for LRW
python test.py \
--lrw 'enter_data_path' \
--checkpoint ./data/Pretrained_Ckpt.ckpt \
--batch_size 80
--radius 16 --slot 112 --head 8 \
--test_aug \
--gpu 0
```

|       Architecture      |   Acc.   |
|:-----------------------:|:--------:|
|Resnet18 + MS-TCN + Multi-head Visual-audio Mem   |   88.508   |

## Training the Model
To train the model, run following command:
```shell
# One GPU Training example for LRW
python train.py \
--lrw 'enter_data_path' \
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 40 \
--radius 16 --n_slot 112 --head 8 \
--augmentations \
--mixup \
--gpu 0
```

```shell
# Dataparallel GPU Training example for LRW
python train.py \
--lrw 'enter_data_path' \
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 80 \
--radius 16 --n_slot 112 --head 8 \
--augmentations \
--mixup \
--dataparallel \
--gpu 0,1
```

```shell
# Distributed Dataparallel GPU Training example for LRW
python -m torch.distributed.launch --nproc_per_node='# of GPU' train.py \
--lrw 'enter_data_path' \
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 40 \
--radius 16 --n_slot 112 --head 8 \
--augmentations \
--mixup \
--distributed \
--gpu 0,1,2,3
```

Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint_dir`: the location for saving checkpoint
- `--checkpoint`: the checkpoint file
- `--batch_size`: batch size
- `--augmentation`: whether performing augmentation  
- `--distributed`: Use DataDistributedParallel  
- `--dataparallel`: Use DataParallel
- `--mixup`: Use mixup augmentation 
- `--gpu`: gpu for using `--lr`: learning rate `--n_slot`: memory slot size `--radius`: scaling factor for addressing score `--head`: number of heads for visual-audio memory
- Refer to `train.py` for the other training parameters

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{kim2022distinguishing,
  title={Distinguishing Homophenes using Multi-head Visual-audio Memory for Lip Reading},
  author={Kim, Minsu and Yeo, Jeong Hun and Ro, Yong Man},
  booktitle={Proceedings of the 36th AAAI Conference on Artificial Intelligence, Vancouver, BC, Canada},
  volume={22},
  year={2022}
}
```
