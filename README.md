# Lip to Speech Synthesis with Visual-Context-Attentional-GAN

This repository contains the PyTorch implementation of the following paper:
> **Lip to Speech Synthesis with Visual-Context-Attentional-GAN**<br>
> Minsu Kim, Joanna Hong, and Yong Man Ro<br>
> Paper: https://proceedings.neurips.cc/paper/2021/file/16437d40c29a1a7b1e78143c9c38f289-Paper.pdf<br>

<div align="center"><img width="90%" src="img/Img.png?raw=true" /></div>

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
After download the dataset, preprocess the dataset with the following scripts.<br>
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
`Preprocess.py` align faces and generate videos. <br>
```shell
python Preprocess.py \
--Data_dir "Data dir of extracted images and audio of GRID_corpus" \
--Landmark "Downloaded landmark dir of GRID" \
--Output_dir "Output dir of processed data"
```

## Training the Model
To train the model, run following command:

```shell
# Data Parallel training example using 4 GPUs for multi-speaker setting in GRID
python train.py \
--grid 'enter_the_processed_data_path' \
--checkpoint_dir 'enter_the_path_for_save' \
--batch_size 88 \
--epochs 500 \
--subject 'overlap' \
--eval_step 720 \
--dataparallel True \
--gpu 0,1,2,3
```

```
# 1 GPU training example for GRID for multi-speaker setting in GRID
python train.py \
--grid 'enter_the_processed_data_path' \
--checkpoint_dir 'enter_the_path_for_save' \
--batch_size 22 \
--epochs 500 \
--subject 'overlap' \
--eval_step 1000 \
--dataparallel False \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--grid`: Dataset location (grid)
- `--checkpoint_dir`: directory for saving checkpoints
- `--checkpoint` : saved checkpoint where the training is resumed from
- `--batch_size`: batch size 
- `--epochs`: number of epochs 
- `--mode`: train / val / test
- `--augmentations`: whether performing augmentation
- `--dataparallel`: Use DataParallel
- `--subject`: different speaker settings, s# is speaker specific training, 'overlap' for multi-speaker setting, 'unseen' for unseen-speaker setting, 'four' for four speaker training
- `--gpu`: gpu number for training
- `--lr`: learning rate
- `--eval_step`: steps for performing evaluation
- `--window_size`: number of frames to be used for training
- Refer to `train.py` for the other training parameters

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
```
Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint`: the checkpoint file
- `--batch_size`: batch size  `--mode`: train / val / test
- `--test_aug`: whether performing test time augmentation  `--distributed`: Use DataDistributedParallel  `--dataparallel`: Use DataParallel
- `--gpu`: gpu for using `--lr`: learning rate `--n_slot`: memory slot size `--radius`: scaling factor for addressing score
- Refer to `test.py` for the other testing parameters

## Pretrained Models
You can download the pretrained models. <br>
Put the ckpt in './data/'

**Bi-GRU Backend**
- https://drive.google.com/file/d/1wkgkRWxu7JM0uaNHmcyCpvVz9OFar8Do/view?usp=sharing <br>

To test the pretrained model, run following command:
```shell
# Testing example for LRW
python main.py \
--lrw 'enter_data_path' \
--checkpoint ./data/GRU_Back_Ckpt.ckpt \
--batch_size 80 --backend GRU\
--mode test --radius 16 --n_slot 88 \
--test_aug True --distributed False --dataparallel False \
--gpu 0
```

**MS-TCN Backend**
- https://drive.google.com/file/d/1uHZbmk9fgMqYVfvaoMUe-9XlGvQnXEcS/view?usp=sharing

To test the pretrained model, run following command:
```shell
# Testing example for LRW
python main.py \
--lrw 'enter_data_path' \
--checkpoint ./data/MSTCN_Back_Ckpt.ckpt \
--batch_size 80 --backend MSTCN\
--mode test --radius 16 --n_slot 168 \
--test_aug True --distributed False --dataparallel False \
--gpu 0
```

|       Architecture      |   Acc.   |
|:-----------------------:|:--------:|
|Resnet18 + MS-TCN + Multi-modal Mem   |   85.864    |
|Resnet18 + Bi-GRU + Multi-modal Mem   |   85.408    |

## AVSR
You can also use the pre-trained model to perform Audio Visual Speech Recognition (AVSR), since it is trained with both audio and video inputs. <br>
In order to use AVSR, just use ''tr_fusion'' (refer to the train code) for prediction.

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{kim2021multimodalmem,
  title={Multi-Modality Associative Bridging Through Memory: Speech Sound Recollected From Face Video},
  author={Kim, Minsu and Hong, Joanna and Park, Se Jin and Ro, Yong Man},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={296--306},
  year={2021}
}
```
