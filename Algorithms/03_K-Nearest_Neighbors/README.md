# Face classification with KNN

## Overview
This directory implements a **K-Nearest Neighbour** model to classify the faces in the LFW dataset. 

# Pipeline
1. Load dataset (use only faces, which has more than 10 training samples)
2. Flatten the images
3. Use PCA embedding on flatten images
4. 80/20 split of embeddings
5. Train KNN model


### Diagram
TODO

### Output Formula
TODO

## Key Details


| Parameter                         | Value                               |
|-----------------------------------|-------------------------------------|
| **Algorithm**                     | K-Nearest Neighbors                 |
| **Learning Type**                 | Supervised Learning                 |
| **Dataset**                       | Labelled Faces in the Wild - Kaggle |
| **Data Split**                    | 80/20 Split                         |
| **Test Dataset Accuracy (80/20)** | **21.16 %**                         |



## TODO
1. Instead, of PCA use FaceNet/ArcFace for embedding faces.   

