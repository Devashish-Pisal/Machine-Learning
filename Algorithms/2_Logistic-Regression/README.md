# Spam Message Datection

## Overview
This directory implements a **Logistic Regression** model to classify spam messages using the **SMS Spam Collection Dataset - Kaggle**. The model is trained to minimize the **Binary Class Entropy (BCE)** loss, achieving convergence at a test loss of **0.19** after **1000 epochs**.

## Model Architecture
The model consists of a **single linear layer** that takes text message as input vectorizes it, followed by a sigmoid activation for binary classification.

### Diagram
TODO

### Output Formula
TODO

## Key Details

| Parameter                  | Value                                |
|----------------------------|--------------------------------------|
| **Algorithm**              | Logistic Regression                  |
| **Dataset**                | SMS Spam Collection Dataset - Kaggle |
| **Data Split**             | 80% Training, 20% Testing            |
| **Loss Function**          | Binary Class Entropy                 |
| **Train-Loss Convergence** | 0.15 (after 1000 epochs)             |
| **Test Loss**              | 0.19                                 |


## TODO
1. Use Sigmoid + BCE together with `torch.nn.BCEWithLogistLoss()` 
2. Use weight decay (regularization)