# California Housing Price Prediction with Linear Regression

## Overview
This directory implements a **Linear Regression** model to predict housing prices using the **California Housing Dataset**. The model is trained to minimize the **Mean Squared Error (MSE)** loss, achieving convergence at a loss of **0.55** after **1000 epochs**.


## Model Architecture
The model consists of a **single linear layer** that maps 8 input features to 1 output.

### Diagram

TODO

### Output Formula

TODO


## Key Details

| Parameter          | Value                        |
|--------------------|------------------------------|
| **Algorithm**      | Linear Regression            |
| **Dataset**        | California Housing Dataset    |
| **Data Split**     | 80% Training, 20% Testing     |
| **Loss Function**  | Mean Squared Error (MSE)      |
| **Convergence**    | Loss: 0.55 (after 1000 epochs)|

## TODO
1. Use Early stopping in training loop to save time and avoid overfitting. 
2. Experiment with different optimizers and learning rates.