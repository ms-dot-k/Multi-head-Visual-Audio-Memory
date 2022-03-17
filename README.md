# Lip to Speech Synthesis with Visual Context Attentional GAN

This repository contains the PyTorch implementation of the following paper:
> **Lip to Speech Synthesis with Visual Context Attentional GAN**<br>
> Minsu Kim, Joanna Hong, and Yong Man Ro<br>
> \[[Paper](https://proceedings.neurips.cc/paper/2021/file/16437d40c29a1a7b1e78143c9c38f289-Paper.pdf)\] \[[Demo Video](https://drive.google.com/file/d/1jkD6KAMYQ7BPD_Z8oiwhmkA-wa-UcaYF/view?usp=sharing)\]

<div align="center"><img width="75%" src="img/Img.PNG?raw=true" /></div>

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
- librosa
- pystoi
- pesq
- scipy

### Datasets
#### Download
GRID dataset (video normal) can be downloaded from the below link.
- http://spandh.dcs.shef.ac.uk/gridcorpus/

For data preprocessing, download the face landmark of GRID from the below link. 
- https://drive.google.com/file/d/1MDLmREuqeWin6CituMn4Z_dhIIJAwDGo/view?usp=sharing

#### Preprocessing
After download the dataset, preprocess the dataset with the following scripts in `./preprocess`.<br>
It supposes the data directory is constructed as
```
Data_dir
├── subject
|   ├── video
|   |   └── xxx.mpg
```

1. Extract frames <br>
`Extract_frames.py` extract images and audio from the video. <br>
```shell
python Extract_frames.py --Grid_dir "Data dir of GRID_corpus" --Out_dir "Output dir of images and audio of GRID_corpus"
```

2. Align faces and audio processing <br>
`Preprocess.py` aligns faces and generates videos, which enables cropping the video lip-centered during training. <br>
```shell
python Preprocess.py \
--Data_dir "Data dir of extracted images and audio of GRID_corpus" \
--Landmark "Downloaded landmark dir of GRID" \
--Output_dir "Output dir of processed data"
```

## Training the Model
The speaker setting (different subject) can be selected by `subject` argument. Please refer to below examples. <br>
To train the model, run following command:

```shell
# Data Parallel training example using 4 GPUs for multi-speaker setting in GRID
python train.py \
--grid 'enter_the_processed_data_path' \
--checkpoint_dir 'enter_the_path_to_save' \
--batch_size 88 \
--epochs 500 \
--subject 'overlap' \
--eval_step 720 \
--dataparallel \
--gpu 0,1,2,3
```

```shell
# 1 GPU training example for GRID for unseen-speaker setting in GRID
python train.py \
--grid 'enter_the_processed_data_path' \
--checkpoint_dir 'enter_the_path_to_save' \
--batch_size 22 \
--epochs 500 \
--subject 'unseen' \
--eval_step 1000 \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--grid`: Dataset location (grid)
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--epochs`: number of epochs 
- `--augmentations`: whether performing augmentation
- `--dataparallel`: Use DataParallel
- `--subject`: different speaker settings, `s#` is speaker specific training, `overlap` for multi-speaker setting, `unseen` for unseen-speaker setting, `four` for four speaker training
- `--gpu`: gpu number for training
- `--lr`: learning rate
- `--eval_step`: steps for performing evaluation
- `--window_size`: number of frames to be used for training
- Refer to `train.py` for the other training parameters

The evaluation during training is performed for a subset of the validation dataset due to the heavy time costs of waveform conversion (griffin-lim). <br>
In order to evaluate the entire performance of the trained model run the test code (refer to "Testing the Model" section).

### check the training logs
```shell
tensorboard --logdir='./runs/logs to watch' --host='ip address of the server'
```
The tensorboard shows the training and validation loss, evaluation metrics, generated mel-spectrogram, and audio


## Testing the Model
To test the model, run following command:
```shell
# Dataparallel test example for multi-speaker setting in GRID
python test.py \
--grid 'enter_the_processed_data_path' \
--checkpoint 'enter_the_checkpoint_path' \
--batch_size 100 \
--subject 'overlap' \
--save_mel \
--save_wav \
--dataparallel \
--gpu 0,1
```

Descriptions of training parameters are as follows:
- `--grid`: Dataset location (grid)
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--dataparallel`: Use DataParallel
- `--subject`: different speaker settings, `s#` is speaker specific training, `overlap` for multi-speaker setting, `unseen` for unseen-speaker setting, `four` for four speaker training
- `--save_mel`: whether to save the 'mel_spectrogram' and 'spectrogram' in `.npz` format
- `--save_wav`: whether to save the 'waveform' in `.wav` format
- `--gpu`: gpu number for training
- Refer to `test.py` for the other parameters

## Test Automatic Speech Recognition (ASR) results of generated results: WER
Transcription (Ground-truth) of GRID dataset can be downloaded from the below link.
- https://drive.google.com/file/d/1q_v4acR_xsHb75P09jKAAtNONVo35ueR/view?usp=sharing

move to the ASR_model directory
```shell
cd ASR_model/GRID
```

To evaluate the WER, run following command:
```shell
# test example for multi-speaker setting in GRID
python test.py \
--data 'enter_the_generated_data_dir (mel or wav) (ex. ./../../test/spec_mel)' \
--gtpath 'enter_the_downloaded_transcription_path' \
--subject 'overlap' \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--data`: Data for evaluation (wav or mel(.npz))
- `--wav` : whether the data is waveform or not
- `--batch_size`: batch size 
- `--subject`: different speaker settings, `s#` is speaker specific training, `overlap` for multi-speaker setting, `unseen` for unseen-speaker setting, `four` for four speaker training
- `--gpu`: gpu number for training
- Refer to `./ASR_model/GRID/test.py` for the other parameters


### Pre-trained ASR model checkpoint
Below lists are the pre-trained ASR model to evaluate the generated speech. <br>
WER shows the original performances of the model on ground-truth audio.

|       Setting       |   WER   |
|:-------------------:|:--------:|
|GRID (constrained-speaker) |   [0.83 %](https://drive.google.com/file/d/1i73hVfC78r07EwKfnNtt8_vfMR9w51vx/view?usp=sharing)  |
|GRID (multi-speaker)       |   [0.37 %](https://drive.google.com/file/d/14vKWXS22vKKzxPN2Uc-IXRQc-WQwbQ-1/view?usp=sharing)  |
|GRID (unseen-speaker)      |   [1.67 %](https://drive.google.com/file/d/17EAciXs6xUzI80OyB0_ekjHzJmg4pQ1b/view?usp=sharing)  |
|LRW                        |   [1.54 %](https://drive.google.com/file/d/1F6FozeiSbZ_HjqqA-W4-qjFvWrjCHLrG/view?usp=sharing)  |

Put the checkpoints in `./ASR_model/GRID/data` for GRID, and in `./ASR_model/LRW/data` for LRW.

## Citation
If you find this work useful in your research, please cite the paper:
```
@article{kim2021vcagan,
  title={Lip to Speech Synthesis with Visual Context Attentional GAN},
  author={Kim, Minsu and Hong, Joanna and Ro, Yong Man},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
