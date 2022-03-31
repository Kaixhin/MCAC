Minimal Criterion Artist Collective
===================================
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Usage
-----

Run `main.py` with a choice of generator `--generator StyleGAN|CPPN|FractalFlame`. Requires [PyTorch](https://pytorch.org/). Details on the algorithm are available in the [paper](https://github.com/Kaixhin/MCAC/blob/master/paper.pdf).

[`libmypaint`](https://github.com/Kaixhin/MCAC/tree/mypaint) and [MIDI](https://github.com/Kaixhin/MCAC/tree/midi) generators are available under their corresponding branches (undocumented).

Samples
-------

Image and audio samples can be found in [samples](https://github.com/Kaixhin/MCAC/tree/master/samples) folder.

| | | | |
|-|-|-|-|
| ![](https://github.com/Kaixhin/MCAC/blob/master/samples/StyleGAN/274.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/StyleGAN/1013.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/StyleGAN/200.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/StyleGAN/287.png) |
| ![](https://github.com/Kaixhin/MCAC/blob/master/samples/CPPN/1862.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/CPPN/1871.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/CPPN/2020.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/CPPN/2033.png) |
| ![](https://github.com/Kaixhin/MCAC/blob/master/samples/libmypaint/1405.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/libmypaint/2318.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/libmypaint/2421.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/libmypaint/2587.png) |
| ![](https://github.com/Kaixhin/MCAC/blob/master/samples/FractalFlame/201.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/FractalFlame/16.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/FractalFlame/60.png) | ![](https://github.com/Kaixhin/MCAC/blob/master/samples/FractalFlame/58.png) |

Citation
--------

```tex
@inproceedings{arulkumaran2022minimal,
  author = {Arulkumaran, Kai and Nguyen-Phuoc, Thu},
  title = {Minimal Criterion Artist Collective},
  booktitle = {Genetic and Evolutionary Computation Conference Companion},
  year = {2022}
}
```
