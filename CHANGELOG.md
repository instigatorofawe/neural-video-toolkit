# Changelog

## [0.3.2] - 2021-02-04

### Fixed

- Removed hard-coded argument from upscale.py

## [0.3.1] - 2021-02-04

### Added

- Added duplicate frame detection so as to minimize unnecessary computation. This increases performance significantly.

## [0.3.0] - 2021-02-04

### Added

- Image super resolution using Content Adaptive Resampler. Code modified from [Wanjie Sun, Zhenzhong Chen. "Learned
  Image Downscaling for Upscaling using Content Adaptive Resampler". arXiv preprint arXiv:1907.12904, 2019](
  https://github.com/sunwj/CAR)

## [0.2.2] - 2021-01-31

### Changed

- Changed default compression settings to H.265 and pixel format to yuv420p10le (10-bit)
- Changed color matrix of encoding to BT709

## [0.2.1] - 2021-01-31

### Fixed

- Fixed loading and data processing for subpixel CNN.

## [0.2.0] - 2021-01-31

### Added

- Image super resolution using subpixel CNN. Updated documentation. Currently supports single channel models. Code
  modified from [Keras example: Image Super-Resolution using an Efficient Sub-Pixel CNN](
  https://github.com/keras-team/keras-io/blob/master/examples/vision/super_resolution_sub_pixel.py). Methodology
  from [W. Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
  Neural Network, Computer Vision and Pattern Recognition*, **2016**](https://arxiv.org/abs/1609.05158).

## [0.1.2] - 2021-01-31

### Fixed

- Fixed parsing of command line arguments

### Changed

- Code refactorizations for style and readability

## [0.1.1] - 2021-01-30

### Fixed

- File inputs were hard-coded in upscale.py; they are properly parsed from the command line now

## [0.1.0] - 2021-01-30

### Added

- README containing installation instructions, usage guide, and documentation of supported model architectures
- MIT license
- CHANGELOG
- Package installation scripts
- Skeleton code for neural upscaling
- ESRGAN code modified from [X. Wang et al., Enhanced Super-Resolution Generative Adversarial Networks, *The European
  Conference on Computer Vision Workshops (ECCVW)*, **2018**](https://github.com/BlueAmulet/ESRGAN)

[0.1.0]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.1.0
[0.1.1]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.1.1
[0.1.2]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.1.2
[0.2.0]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.2.0
[0.2.1]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.2.1
[0.2.2]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.2.2
[0.3.0]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.3.0
[0.3.1]:https://github.com/instigatorofawe/neural-video-toolkit/releases/tag/0.3.1
