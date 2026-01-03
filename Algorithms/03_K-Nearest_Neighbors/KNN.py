'''
Pipeline:
1. Load All Data
2. Drop data with low training samples
3. Convert image data to embedding (for better training)
4. train KNN model
5. Calculate accuracies for 2-Way split (if we use k-fold validation, it will take too much time since all process is running on CPU)
'''

import pandas as pd
import os
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load target classes
all_names_df = pd.read_csv("E:\\(_Coding_Data_)\\(_Github_Repositories_)\\Machine-Learning\\Datasets\\Labelled_Faces_in_the_Wild\\lfw_allnames.csv")
print(f"Total classes in the dataset: {all_names_df.shape[0]}")
print(f"Total images in the dataset: {all_names_df['images'].sum()}")
removed_names_df = all_names_df[all_names_df['images'] < 10]
print(f"Removed classes with less than 10 training smaples: {removed_names_df.shape[0]}")
print(f"Removed images with less than 10 training samples: {removed_names_df['images'].sum()}")
reduced_names_df = all_names_df[all_names_df['images'] >= 10]
print(f"Kept classes with more than 10 training samples: {reduced_names_df.shape[0]}")
print(f"Kept images with more than 10 training samples: {reduced_names_df['images'].sum()}")
print("=====================================================================================")

# Assign integer value to the labels
# print(reduced_names_df.duplicated().sum())  # Check for no duplicate labels
labels = reduced_names_df['name'].tolist()



# flatten the images
flatten_images_ary = []
int_labels_ary = []
for label in labels:
    base_folder_path = f"E:\\(_Coding_Data_)\\(_Github_Repositories_)\\Machine-Learning\\Datasets\\Labelled_Faces_in_the_Wild\\lfw-deepfunneled\lfw-deepfunneled\\{label}"
    for filename in os.listdir(base_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image = Image.open(os.path.join(base_folder_path, filename))
            flatten_img = np.array(image).flatten()
            flatten_images_ary.append(flatten_img)
            int_labels_ary.append(labels.index(label))
        else:
            raise ValueError("UNKNOWN IMAGE FORMATE IN THE DATASET!")
print(f"Number of images flattened: {len(flatten_images_ary)}")
print(f"Total number of labels: {len(int_labels_ary)}")
print("=====================================================================================")


# Embedding step: Use PCA on flattened images
pca = PCA(n_components=200, whiten=True) # Reduce images to 100 components
X_pca = pca.fit_transform(flatten_images_ary)
print("PCA embedding finished!")

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X_pca, int_labels_ary, test_size=0.2, stratify=int_labels_ary, random_state=42) # X_train/test are 2D numpy arrays & y_train/test are 1D numpy arrays
print("80/20 split done!")
print("=====================================================================================")


# Train and Test KNN model
model = KNeighborsClassifier(n_neighbors=1, metric="cosine", weights="distance")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy of KNN with 80/20 split: {accuracy_score(y_test,y_pred)*100:.2F}")
print("=====================================================================================")


