import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


#Load data
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target_classes = iris.target_names # ['setosa', 'versicolor', 'virginica'] --> [0,1,2]


# Data is currently in numpy formate
# Convert the data to pandas dataframe
df = pd.DataFrame(
    data=iris.data,
    columns=feature_names,
)
df['target'] = y_iris



#=========================== IMPLEMENTATION WITH 80-20 SPLIT ==========================================================

y = df['target']
X = df[feature_names]


# 80-20 Split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=42)
model_1 = GaussianNB()
model_1.fit(X_train, y_train)
y_pred = model_1.predict(X_test)
print("========================= 80/20 Split Performance ==============================")
report = classification_report(y_test, y_pred, output_dict=True)  # output_dic = True --> Return report as dictionary and not as string
print(classification_report(y_test, y_pred))
print(f"Overall 80/20 Split Accuracy: {report['accuracy']*100:.2f}%")

#Result
'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
'''

#=========================== IMPLEMENTATION WITH CROSS VALIDATION (K-Fold CV) ==========================================================

k = 5
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
folds = np.array_split(shuffled_df, k)
scores = []

print("========================= K-Fold CV Performance (Reliable) ==============================")

# K-fold training and testing
for i in range(k):
    test_df = folds[i]
    train_df = pd.concat([folds[j] for j in range(k) if j != i])
    X_train = train_df[feature_names]
    y_train = train_df['target']
    X_test = test_df[feature_names]
    y_test = test_df['target']

    model_2 = GaussianNB()
    model_2.fit(X_train, y_train)
    y_pred = model_2.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    print(f"Fold {i} accuracy: {accuracy:.2F}")

overall_accuracy = sum(scores)/len(scores)
print(f"Overall Accuracy with K-Fold CV: {overall_accuracy*100:.2F}%")
print("=======================================================================================")

