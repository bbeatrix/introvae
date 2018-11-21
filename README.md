# Tensorflow Implementation of IntroVAE

This repository contains an implementation of the IntroVAE model in tensorflow based on the paper titled "IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis" https://arxiv.org/abs/1807.06358.

**Usage**

***1. Download and extract the CelebA-HQ-128x128 dataset***

The CelebA-HQ-128x128 dataset is a downscaled version of the CelebA-HQ (1024x1024) dataset.

It can be downloaded from the following Dropbox link:
https://www.dropbox.com/s/ayz2roywuq253l0/celebA-HQ-128x128.tar.gz?dl=0

Extract the contents of the above tar.gz in the datasets directory.

```bash
$ mkdir datasets
$ tar xvzf celebA-HQ-128x128.tar.gz -C datasets
```


***2. Train a model***

For a baseline training on the CelebA-HQ-128x128 dataset, simply run the following script.
```bash
$ bash ./baseline_128x128.sh
```
