# IRIS Species Classification

## Overview
This directory implements a **Naive Bayes** model to classify IRIS Species based on their sepal and petal characteristics. There are mainly 3 types of Naive Bayes models. 
1. Gaussian Naive Bayes: Assumes features follow a Gaussian distribution
2. Multinomial Naive Bayes: Suitable for text classification
3. Bernoulli Naive Bayes: Features are binary (0/1)

Note: Currently in this directory only Gaussian NB is implemented.

## Model Architecture
The model is based on **Bayes Theorem**. Model first calculates the prior probabilities then calculates the likelihood and at last calculates the 
posterior probabilities. For classification chooses the class with the highest posterior probability.

### Diagram
TODO

### Output Formula
TODO

## Key Details


| Parameter                         | Value                          |
|-----------------------------------|--------------------------------|
| **Algorithm**                     | Naive Bayes                    |
| **Learning Type**                 | Supervised Learning            |
| **Dataset**                       | IRIS Species Dataset           |
| **Data Split**                    | 80/20 Split + Cross Validation |
| **Test Dataset Accuracy (80/20)** | **100 %**                      |
| **K-Fold CV Accuracy**            | **96 %**                       |



## TODO
1. Implement remaining 2 types of NB models

