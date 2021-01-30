# Neural Video Toolkit

A toolkit for integrating neural video processing methods into an FFMPEG pipeline. Both Tensorflow and PyTorch models
are supported. Additional model architectures can be supported through the extension of this code.

## Contents

1. [Package Requirements](#package-requirements)
1. [Setup](#setup)
1. [Usage Guide](#usage-guide)
    1. [Image super resolution](#image-super-resolution)
    1. [Video input from ffmpeg](#video-input-from-ffmpeg)
1. [Architectures](#architectures)
    1. [ESRGAN](#esrgan)
1. [Changelog](CHANGELOG.md)

## Package Requirements

Depending on your hardware/operating system, you may need to use a different procedure when installing TensorFlow or
PyTorch. This package also requires CUDA 11.0 as well as a compatible GPU. This package was developed on Ubuntu LTS
20.04 with an Nvidia GeForce RTX 3080 (10 GB VRAM).

Package requirements can be installed using pip.

```
pip install tf-nightly-gpu 
```

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

```
pip install ffmpeg-python argparse opencv-python keras
```

## Setup

Package can be installed from source using pip. Navigate to the repository root directory and run the following command.

```
pip install .
```

## Usage Guide

### Image super resolution

Using image super resolution methods, it is possible to scale up video on a frame-by-frame basis. This naive approach is
fairly suitable for animation and cartoons. This script uses ffmpeg-python in order to process a video stream through a
neural image super resolution model.

Usage:

```
usage: upscale.py [-h] [-o output_filename] -m model_filename
                  [-a architecture]
                  input_filename

Upscale a video file using an image super resolution method. Uses ffmpeg for
encoding and decoding.

positional arguments:
  input_filename        Filename of input video file.

optional arguments:
  -h, --help            show this help message and exit
  -o output_filename, --output output_filename
                        Filename of output video file. Defaults to
                        (input_filename)_upscaled.
  -m model_filename, --model model_filename
                        Filename of model weights.
  -a architecture, --arch architecture
                        Architecture for upscaling method. Supported values:
                        esrgan
```

## Architectures

### ESRGAN

Code modified from [X. Wang et al., Enhanced Super-Resolution Generative Adversarial Networks, *The European Conference
on Computer Vision Workshops (ECCVW)*, **2018**](https://github.com/BlueAmulet/ESRGAN)

Pretrained models can be found at [https://upscale.wiki/wiki/Model_Database](https://upscale.wiki/wiki/Model_Database).

## [Changelog](CHANGELOG.md)