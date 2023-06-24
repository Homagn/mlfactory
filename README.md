About
=====
***mlfactory*** is a simple modular wrapper library that abstracts several different types of neural network architectures and techniques for computer vision (pytorch and tensorflow backend) providing seemless easy to use training in a few lines of code. 

Using the standard modular philosophy you can also define your own neural network in pytorch/tensorflow, and if youre lazy to write the data loaders or the training loop then pass the network to our submodules !, or vice versa.



Table of contents
=================

<!--ts-->
   * [Getting Started](#getting-started)
   * [Out of box colab usage](#out-of-box-colab-usage)
<!--te-->


Getting Started
===============

pip install mlfactory



Out of box colab usage
======================

Machine Learning and AI applications full pipeline
--------------------------------------------------

1. High definition mapping using monocular camera (using monocular depth estimation and superglue feature extractor)
- https://colab.research.google.com/drive/1lZYHjYszROIvMjtZgAp7r5eL1YZC9x9M?usp=sharing

2. Simple and fast visual odometry directly from MOV files and output pose trajectory in open3d
- https://colab.research.google.com/drive/1Nr1nYFBKieDQG6UeNsnrV4gHh-l-65G0?usp=sharing

Compose machine learning applications in a modular way
------------------------------------------------------

1. (NYUV2 dataloader) Easy monocular depth estimation 
- https://colab.research.google.com/drive/1T2gONs_gst4zpdS7fBoIaQclgg3J2Jgk?usp=sharing

2. Finetune deeplabv3 for any general binary segmentation using less than 100 samples
- https://colab.research.google.com/drive/1y0wirrjuoha3_SQk52IMJmL99iPGT6DZ?usp=sharing


Annotation and other computer vision utilities
----------------------------------------------

1. Polygon annotation tool allowing to create polygon masks for segmentation directly in colab
- https://colab.research.google.com/drive/1YUoMU3H_m9KM6xTrAKAy9ebiDWWzXTle?usp=sharing


2. Easy usage of superglue neural network based image feature matching
- https://colab.research.google.com/drive/1fqnW1-Dlwz3fYlacTjMUmu5gxreGfjj6?usp=sharing

3. Easy usage of holistically nested edge detection for generic high level edge detection
- https://colab.research.google.com/drive/174STj0gsUZ4qOrbj_WFXhtSfwcef-oMi?usp=sharing








Upcoming
========

- Hollistic edge detection network (HED) - https://github.com/sniklaus/pytorch-hed pair it up with opencv distance transform

- behavior transformers and using GPT - https://github.com/notmahi/bet and https://github.com/notmahi/miniBET

- base transformer model definition + segformer model definitions pytorch

- diffusion models from scratch pytorch

- MIRnet keras for low light image enhancement - https://keras.io/examples/vision/mirnet/

- SRGAN for image superresolution - https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

- differentiable pointcloud generation models - https://github.com/puhsu/point-clouds

- image animation generation models - https://github.com/snap-research/articulated-animation (code cloned in /ml/misctools/articulated-animation/, run the working demo-> demo_outofbox.py)

- FCOSnet pytorch - https://github.com/VectXmy/FCOS.Pytorch

- dataloader for ouster lidar datasets

- Coco bounding box dataloader colab

- tum_rgbd dataloader

- DeepLabv3 finetuning module

- Resnet finetunining module for 2D image regregression (pose estimation example)