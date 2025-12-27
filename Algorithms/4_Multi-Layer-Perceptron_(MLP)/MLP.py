import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import torchviz

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


transform = transforms.ToTensor() # Converts images to tensors

# Downloads the dataset if not available at the root location
train_dataset = MNIST(root='F:\\Tmp\\Machine-Learning\\Datasets', train=True, download=True, transform=transform)
test_dataset = MNIST(root='F:\\Tmp\\Machine-Learning\\Datasets', train=False, download=True, transform=transform)


# Loading data in batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

'''
for images, labels in train_loader:
    print(images.shape)  # -->  torch.Size([64, 1, 28, 28]) --> (64 images in one batch; 1 for grayscale channel, 28*28 pixel size of each image)
    print(labels.shape)  # -->  torch.Size([64]) --> (Per image one label --> 64 labels)
    break
'''

# Define Model class
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__() # Initialize the parent class which nn.Module
        self.layers = nn.Sequential(
            nn.Linear(28*28, 128),  # Layer 1 : 28*28 input features --> 128 Outputs
            nn.ReLU(),
            nn.Linear(128, 64),  # Layer 2 : 128 input features --> 64 outputs
            nn.ReLU(),
            nn.Linear(64, 10) # Layer 3 : 64 inputs --> 10 outputs (since total 10 classes)
        )

    def forward(self, x):
        output = self.layers(x)
        return output


# Create model instance
model = MultiLayerPerceptron()

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# MNIST Dataset or Data loaders cannot be moved to GPU since they are not tensors, but they
# contain tensors (images, labels) which will be moved to GPU in training and testing loops.


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Data
epochs = 20
training_loss = np.zeros(epochs)

# Train the model
model.train()
for epoch in range(epochs):
    total_batch_loss = 0.0
    for images, labels in train_loader:
        # Make 2D images 1D --> Flatten images
        images = images.view(images.size(0), -1)

        # Move data to GPU
        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)
        loss = criterion(prediction, labels)
        total_batch_loss += loss.item()
        last_seen_lost = loss.item()
        optimizer.zero_grad() # Make initial gradients zero
        loss.backward() # Backpropagation
        optimizer.step() # Update Weights

    average_batch_loss = total_batch_loss / len(train_loader)
    training_loss[epoch] = average_batch_loss
    print(f'Training Loss for epoch {epoch + 1}: {average_batch_loss:.4F}')

# Plot loss graph
'''
plt.scatter(range(1,epochs+1), training_loss, s=5)
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.title('Training Loss vs. Epochs')
plt.show()
'''


# Testing
model.eval()
with torch.no_grad():
    correct_classified = 0
    incorrect_classified = 0
    for images, labels in test_loader:
        images = images.view(images.size(0), -1)
        images = images.to(device)
        labels = labels.to(device)
        prediction = model(images)

        # Compare prediction
        pred_classes = torch.argmax(prediction, dim=1)
        correct = (pred_classes == labels).sum().item() # Compare with actual label --> assign 0/1s --> sum it --> convert to python number
        incorrect = labels.size(0) - correct

        # Update
        correct_classified += correct
        incorrect_classified += incorrect

    accuracy = correct_classified * 100/ (correct_classified + incorrect_classified)
    print("=================================================")
    print(f"Test Accuracy: {accuracy:.2F}")
    print("=================================================")


'''
# Manaul Testing : In order to manual testing work properly, comment out the TESTING CODE above. 
model.eval()
with torch.no_grad():
    index = 677 # Choose random index for selection of image from test dataset
    image, label = test_dataset[index]
    image = image.view(image.size(0), -1).to(device)
    prediction = model(image)
    prediction = torch.argmax(prediction, dim=1)
    print(f"Predicted Label: {prediction.item()}")
    print(f"Actual Label: {label}")

    actual_image = Image.open("1_Manual-Testing_Black-Background.jpg")
    # actual_image.show("Original image")
    actual_image = actual_image.convert("L") # Convert to gray-scale
    # actual_image.show("Grayscale image")
    actual_image = actual_image.resize((28,28)) # Resize to 28*28 pixels
    # actual_image.show("Resized image")
    actual_image_tensor = transform(actual_image) # Transform image to tensor
    actual_image_flat = actual_image_tensor.view(1,-1).to(device) # Flatten the image and shift to GPU
    prediction_2 = model(actual_image_flat)
    prediction_2 = torch.argmax(prediction_2, dim=1)
    print(f"Actual Image Predicted Label: {prediction_2.item()}")
'''



