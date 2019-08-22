# IntroVAE-PyTorch

### Simple implementation of IntroVAE toward MNIST

![Github](https://img.shields.io/badge/PyTorch-v0.4.1-red.svg?style=for-the-badge&logo=data:image/png)
![Github](https://img.shields.io/badge/python-3.6.3-green.svg?style=for-the-badge&logo=python)

<p align="center">
  <img src="https://github.com/SunnerLi/IntroVAE-PyTorch/blob/master/Img/train_process.gif" width=384 height=384/>
</p> 

Abstract
---
This is the simple implementation of the [paper](https://arxiv.org/abs/1807.06358)- IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis. We only test the idea on MNIST. The LSUM and CelebA example can be found in other [implementation](https://github.com/dragen1860/IntroVAE-Pytorch).

Usage
---
* Train the model for 100 epoch:
```
$ python3 main.py --epochs 100
```
* Sample for 20 digits images
```
$ python3 main.py --n 20
```

Result
---
![](https://github.com/SunnerLi/IntroVAE-PyTorch/blob/master/Img/loss_curve.png)

The above figure demonstrates the loss curve during IntroVAE training. We only test on the case whose size is 64x64. 