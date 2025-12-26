import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torchviz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torchsummary import summary
from sklearn.preprocessing import StandardScaler

# Find the dataset file path
cwd = os.getcwd() # current working directory --> Logistic-Regression
parent1_wd = os.path.dirname(cwd) # Parent directory of current directory --> Algorithms
parent2_wd = os.path.dirname(parent1_wd) # --> Machine-Learning
csv_path = os.path.join(parent2_wd, 'Datasets', 'spam.csv')

# Load dataset
dataframe = pd.read_csv(csv_path, encoding='latin1')
ds = dataframe[['v1', 'v2']]
ds = ds.rename(columns={'v1': 'TARGET', 'v2': 'FEATURE'})

# Convert text data into numerical data
ds['TARGET'] = ds['TARGET'].map({'ham': 0, 'spam': 1})

# Split data into training and testing dataset and vectorize
X_train, X_test, Y_train, Y_test = train_test_split(ds['FEATURE'],ds['TARGET'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) # Term Frequency-Inverse Document Frequency
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Convert data to tensors
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1,1)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).view(-1,1)

# Define the model
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, out_features=1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Create model
input_dim = X_train_tensor.shape[1]
model = LogisticRegression(input_dim)

# Define Loss function and optimizer
criterion = torch.nn.BCELoss() # Binary class entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Do computations on cuda (if installed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)
model = model.to(device)

# Train the model
epochs = 1000
training_loss = torch.zeros(epochs)
model.train()
for epoch in range(epochs):
    output = model(X_train_tensor)
    loss = criterion(output, Y_train_tensor)
    print(f'Training Loss for epoch {epoch + 1}: {loss.item():.2F}')
    training_loss[epoch] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot Loss vs Epochs
plt.scatter(range(1,epochs+1), training_loss, s=5, color='red')
plt.xlabel('Epochs')
plt.ylabel('Training Loss (BCE)')
plt.show()

# Testing model
model.eval()
with torch.no_grad():
    output = model(X_test_tensor)
    test_loss = criterion(output, Y_test_tensor)
    print("================================================")
    print(f'Test Loss: {test_loss.item():.4F}')
    print("================================================")


# Check model manually
with torch.no_grad():
    spam_msg = "Congratulations you have been selected for an exclusive offer claim your reward now today online"
    ham_msg = "Hi, just checking in to see if you’re free to study together after school today."
    spam_msg_vec = vectorizer.transform([spam_msg])
    ham_msg_vec = vectorizer.transform([ham_msg])
    spam_msg_tensor = torch.tensor(spam_msg_vec.toarray(), dtype=torch.float32)
    ham_msg_tensor = torch.tensor(ham_msg_vec.toarray(), dtype=torch.float32)
    spam_msg_tensor = spam_msg_tensor.to(device)
    ham_msg_tensor = ham_msg_tensor.to(device)
    output1 = model(spam_msg_tensor)
    output2 = model(ham_msg_tensor)
    print('================================================')
    print('Testing Spam and Ham Messages Manually')
    print(f'Spam message confidence (> 0.5): {output1.item():.2F}')
    print(f'Ham message confidence (< 0.5) : {output2.item():.2F}')
    print('================================================')




# Visualize the backpropagation control flow
dummy_spam_msg = "Congratulations you have been selected for an exclusive offer claim your reward now today online"
dummy_spam_msg_vec = vectorizer.transform([spam_msg])
dummy_spam_msg_tensor = torch.tensor(spam_msg_vec.toarray(), dtype=torch.float32)
dummy_spam_msg_tensor = dummy_spam_msg_tensor.to(device)
torchviz.make_dot(model(dummy_spam_msg_tensor), params=dict(model.named_parameters())).render("logistic_regression_backward-pass", format="png")

# Print the model summary
print(summary(model, dummy_spam_msg_tensor.shape))