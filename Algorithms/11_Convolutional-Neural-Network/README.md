# CIFAR-100 Image Classification

## Overview
This directory implements a **Convolutional Neural Network** model to classify the images from CIFAR-100 Dataset. 

## Model Architecture
The model consists of **Convolutional Network** and **Fully Connected Network** with ReLU activations. 
The convolution network consist of three blocks each block contain two convolutional layers and one max-pooling layer.
And the fully connected network contains one adaptive average pooling layer and one linear layer.

### Diagram
TODO

### Output Formula
TODO

## Key Details

| Parameter                 | Value                                                                |
|---------------------------|----------------------------------------------------------------------|
| **Algorithm**             | Convolutional Neural Network                                         |
| **Network Type**          | Feedforward Network                                                  |
| **Learning Type**         | Supervised Learning                                                  |
| **Dataset**               | CIFAR-100 Dataset                                                    |
| **Data Split**            | 40000 Training Images, 10000 Validation Images, 10000 Testing Images |
| **Data Augmented**        | Yes                                                                  |
| **Activation Function**   | Convolution Network and FC Network - ReLu                            |
| **Loss Function**         | Cross-Entropy Loss                                                   |
| **Early Stopping?**       | Yes                                                                  |
| **Train-Loss**            | 1.23 (after 59 epochs)                                               |
| **Validation-Loss**       | 1.65 (after 59 epochs)                                               |
| **Test-Loss**             | 1.53 (after 59 epochs)                                               |
| **Test Dataset Accuracy** | **59 %**                                                             |


## TODO
1. Reduce batch (to 512 or 256) and then retrain (because current batch_size is 1024. With large batch sizes model trains fast
but generalizes worse) 
2. Implement LR decay with scheduling.
