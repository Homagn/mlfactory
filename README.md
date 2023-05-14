About
=====
***mlfactory*** is a simple modular wrapper library that abstracts several different types of neural network architectures and techniques for computer vision (pytorch and tensorflow backend) providing seemless easy to use training in a few lines of code. 

Using the standard modular philosophy you can also define your own neural network in pytorch/tensorflow, and if youre lazy to write the data loaders or the training loop then pass the network to our submodules !, or vice versa.



Table of contents
=================

<!--ts-->
   * [Out of box colab usage](#out-of-box-colab-usage)
   * [Getting Started](#getting-started)
<!--te-->

Out of box colab usage
======================

1. (NYUV2 dataloader) Easy monocular depth estimation - https://colab.research.google.com/drive/1T2gONs_gst4zpdS7fBoIaQclgg3J2Jgk?usp=sharing


Getting Started
===============

pip install mlfactory


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