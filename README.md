# Lip to Speech Synthesis with Visual Context Attentional GAN

This repository contains the PyTorch implementation of the following paper:
> **Lip to Speech Synthesis with Visual Context Attentional GAN**<br>
> Minsu Kim, Joanna Hong, and Yong Man Ro<br>
> \[[Paper](https://proceedings.neurips.cc/paper/2021/file/16437d40c29a1a7b1e78143c9c38f289-Paper.pdf)\] \[[Demo Video](https://kaistackr-my.sharepoint.com/:v:/g/personal/ms_k_kaist_ac_kr/EQp2Zao1ZQFDm9xDVuZubKIB_ns_6gk0L6LB3U5jd4jYKw?e=Qw8ddt)\]

<div align="center"><img width="75%" src="img/Img.PNG?raw=true" /></div>

## Requirements
- python 3.7
- pytorch 1.6 ~ 1.8
- torchvision
- torchaudio
- ffmpeg
- av
- tensorboard
- scikit-image 0.17.0 ~
- opencv-python 3.4 ~
- pillow
- librosa
- pystoi
- pesq
- scipy

## GRID
Please refer [here](README_GRID.md) to run the code on GRID dataset.

## LRS2/LRS3
Please refer [here](README_LRS.md) to run the code and model on LRS2 and LRS3 datasets.

## Citation
If you find this work useful in your research, please cite the papers:
```
@article{kim2021vcagan,
  title={Lip to Speech Synthesis with Visual Context Attentional GAN},
  author={Kim, Minsu and Hong, Joanna and Ro, Yong Man},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{kim2023lip,
  title={Lip-to-Speech Synthesis in the Wild with Multi-task Learning},
  author={Kim, Minsu and Hong, Joanna and Ro, Yong Man},
  journal={arXiv preprint arXiv:2302.08841},
  year={2023}
}
```
