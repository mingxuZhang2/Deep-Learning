# Deep-Learning

## 1. Introduction
This repository contains the final project of the course "Deep Learning" in Computer Science at the University of NanKai. The project consists of the implementation of a deep learning model to solve a classification problem. The dataset used is the CIFAR-100.

## 2. CIFAR-100 Dataset
The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 50000 training images and 10000 test images. There are figures of each classes ![CIFAR-100 Example](/fig/cifar100_example.png)

## 3. Model

### Baseline

In previous assignment, we have already implement a simple ResNet model with residual block. The model is trained on CIFAR-10 dataset. In this project, we will use the same model to train on CIFAR-100 dataset. The model is shown as follows:

| Layer Type       | Output Channels | Kernel Size | Stride | Padding | Use Bias | Notes                      |
|------------------|-----------------|-------------|--------|---------|----------|----------------------------|
| **Conv2d**       | 32              | 3 x 3       | 1      | 1       | No       | Initial convolution layer  |
| **BatchNorm2d**  | 32              | -           | -      | -       | -        |                            |
| **ResidualBlock**| -               | -           | -      | -       | -        | Repeated blocks vary below |
| *Conv2d*         | 32              | 3 x 3       | 1      | 1       | No       | Inside Residual Blocks     |
| *BatchNorm2d*    | 32              | -           | -      | -       | -        |                            |
| *Conv2d*         | 32              | 3 x 3       | 1      | 1       | No       | Inside Residual Blocks     |
| *BatchNorm2d*    | 32              | -           | -      | -       | -        |                            |
| **Layer2**       | -               | -           | -      | -       | -        | More Residual Blocks       |
| *Conv2d*         | 64              | 3 x 3       | 2      | 1       | No       | Stride changes to 2        |
| *BatchNorm2d*    | 64              | -           | -      | -       | -        |                            |
| *Conv2d*         | 64              | 3 x 3       | 1      | 1       | No       | Inside Residual Blocks     |
| *BatchNorm2d*    | 64              | -           | -      | -       | -        |                            |
| *Conv2d (shortcut)*| 64           | 1 x 1       | 2      | -       | No       | Shortcut in Layer2         |
| **Layer3**       | -               | -           | -      | -       | -        |                            |
| *Conv2d*         | 128             | 3 x 3       | 2      | 1       | No       |                            |
| *BatchNorm2d*    | 128             | -           | -      | -       | -        |                            |
| **Layer4**       | -               | -           | -      | -       | -        |                            |
| *Conv2d*         | 256             | 3 x 3       | 2      | 1       | No       |                            |
| *BatchNorm2d*    | 256             | -           | -      | -       | -        |                            |
| **Dropout**      | -               | -           | -      | -       | -        | p=0.5                      |
| **Linear**       | 100             | -           | -      | -       | Yes      | Final fully connected layer|
