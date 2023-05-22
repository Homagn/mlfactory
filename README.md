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


Compose machine learning applications in a modular way
------------------------------------------------------

1. (NYUV2 dataloader) Easy monocular depth estimation 
- https://colab.research.google.com/drive/1T2gONs_gst4zpdS7fBoIaQclgg3J2Jgk?usp=sharing


Annotation and other computer vision utilities
----------------------------------------------

1. Polygon annotation tool allowing to create polygon masks for segmentation directly in colab
- https://colab.research.google.com/drive/1YUoMU3H_m9KM6xTrAKAy9ebiDWWzXTle?usp=sharing


2. Easy usage of superglue neural network based image feature matching
- https://colab.research.google.com/drive/1fqnW1-Dlwz3fYlacTjMUmu5gxreGfjj6?usp=sharing










Upcoming
========
0. dataloader for ouster lidar data

1. Coco bounding box dataloader colab

2. tum_rgbd dataloader

3. visual odometry utility (examples/sfm.py and examples/sfm3d/main.py)

4. superglue matcher module

5. deep_modular_reconstruction - 3D scene mapping using superglue and GLPN depth and open3d registration functions

6. DeepLabv3 finetuning module

7. Resnet finetunining module for 2D image regregression (pose estimation example)

8. yolov7 estimation module