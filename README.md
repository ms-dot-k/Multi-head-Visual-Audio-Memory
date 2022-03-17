# Distinguishing Homophenes using Multi-head Visual-audio Memory for Lip Reading

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

## Training the Model
`train.py` saves the weights in `--checkpoint_dir` and shows the training logs in `./runs`.

To train the model, run following command:
```shell
# Distributed training example for LRW
python -m torch.distributed.launch --nproc_per_node='number of gpus' train.py \
--lrw 'enter_data_path' \
--checkpoint_dir 'enter_the_path_for_save' \
--batch_size 55 --epochs 300 \
--radius 16 --slot 112 --head 8 \
--augmentations --mixup \
--distributed \
--gpu 0,1,2,3
```

```shell
# Data Parallel training example for LRW
python train.py \
--lrw 'enter_data_path' \
--checkpoint_dir 'enter_the_path_for_save' \
--batch_size 220 --epochs 300 \
--radius 16 --slot 112 --head 8 \
--augmentations --mixup \
--dataparallel \
--gpu 0,1,2,3
```

Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint_dir`: directory for saving checkpoints
- `--batch_size`: batch size  `--epochs`: number of epochs
- `--augmentations`: whether performing augmentation  `--distributed`: Use DataDistributedParallel  `--dataparallel`: Use DataParallel
- `--gpu`: gpu for using `--lr`: learning rate `--n_slot`: memory slot size `--radius`: scaling factor for addressing score
- `--head`: number of head for the visual-audio memory `--mixup`: whether performing mixup augmentation
- Refer to `main.py` for the other training parameters

### check the training logs
```shell
tensorboard --logdir='./runs/logs to watch' --host='ip address of the server'
```
The tensorboard shows the training and validation loss, evaluation word accuracy


## Testing the Model
To test the model, run following command:
```shell
# Testing example for LRW
python main.py \
--lrw 'enter_data_path' \
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 80 \
--mode test --radius 16 --n_slot 88 \
--test_aug True --distributed False --dataparallel False \
--gpu 0

Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint`: the checkpoint file
- `--batch_size`: batch size  `--mode`: train / val / test
- `--test_aug`: whether performing test time augmentation  `--distributed`: Use DataDistributedParallel  `--dataparallel`: Use DataParallel
- `--gpu`: gpu for using `--lr`: learning rate `--n_slot`: memory slot size `--radius`: scaling factor for addressing score
- Refer to `main.py` for the other testing parameters

## Pretrained Models
You can download the pretrained models. <br>
Put the ckpt in './data/'

**Pretrained model**
- https://drive.google.com/file/d/1YkDb4gcX1UbMTd01uJ24WcUc-VXu0Lgd/view?usp=sharing

To test the pretrained model, run following command:
```shell
# Testing example for LRW
python main.py \
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

## Citation
If you find this work useful in your research, please cite the paper:
```
@article{kim2022distinguishing,
  title={Distinguishing Homophenes using Multi-head Visual-audio Memory for Lip Reading},
  author={Kim, Minsu and Yeo, Jeong Hun and Ro, Yong Man},
  year={2022}
}
```
