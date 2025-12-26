# Handwritten Digits Classification

## Overview
This directory implements a **Multi-Layer Perceptron** model to classify handwritten digits using the MNIST Dataset. The model is trained to minimize the Cross-Entropy loss, achieving convergence at training loss of 0.008 after 20 epochs.

## Model Architecture
The model consists of **three linear layers** with ReLU activations that take flattened 28x28 images as input and output probabilities for 10 digit classes.

### Diagram
TODO

### Output Formula
TODO

## Key Details

| Parameter                  | Value                                       |
|----------------------------|---------------------------------------------|
| **Algorithm**              | Multi-Layer Perceptron                      |
| **Model Type**             | Feedforward Network                         |
| **Learning Type**          | Supervised Learning                         |
| **Dataset**                | MNIST Dataset                               |
| **Data Split**             | 60000 Training Images, 10000 Testing Images |
| **Activation Function**    | Layer 1 & 2 - ReLu                          |
| **Loss Function**          | Cross-Entropy Loss                          |
| **Train-Loss Convergence** | 0.008 (after 20 epochs)                     |
| **Test Dataset Accuracy**  | **97.50 %**                                 |


## TODO
1. Add model visualization code using torchviz.
2. Add model summary code.
