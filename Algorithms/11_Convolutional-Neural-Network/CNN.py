import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

transform = transforms.Compose([transforms.ToTensor()]) # Converts to tensors

# Load training and testing dataset
train_data = CIFAR100(root="F:\\Tmp\\Machine-Learning\\Datasets", train=True, download=True, transform=transform)
test_data = CIFAR100(root="F:\\Tmp\\Machine-Learning\\Datasets", train=False, download=True, transform=transform)

# Load data in batches
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

'''
for images, labels in train_loader:
    print(images.shape)  # -->  torch.Size([64, 3, 32, 32]) --> (64 images in one batch; 3 for RGB channel, 32*32 pixel size of each image)
    print(labels.shape)  # -->  torch.Size([64]) --> (Per image one label --> 64 labels)
    break
'''

# Define Model
class ConvolutionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution Network
        self.conv = nn.Sequential(
            # in_channels = number of input channels --> 3 because of RGB
            # out_channels = number of produced feature maps = number of filters to use = 32
            # kernel_size = filter size = 3x3x3
            # stride = Moves filter 1 pixel at a time
            # padding = 1 pixel border to the input image

            # Layer 1: [Input size - 32x32x3] ---> [Output Size - 16x16x3]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # Depth of output becomes 64
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Generates the output half of the size of input (Abstraction)

            # Layer 2: [Input size - 16x16x3] ---> [Output Size - 8x8x3]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # Depth of output becomes 64
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3: [Input size - 8x8x3] ---> [Output Size - 4x4x3]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=0.2),
        )

        # Neural Network
        self.linear = nn.Sequential(
            nn.Linear(in_features=256*4*4, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=100)
        )

    def forward(self, x):
        conv_output = self.conv(x)
        conv_output = conv_output.view(conv_output.shape[0], -1)
        y_pred = self.linear(conv_output)
        return y_pred

# Define model
model = ConvolutionalNet()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = model.to(device)


# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20


# Store training stats
training_loss = np.zeros(epochs)

# Training
model.train()
for epoch in range(epochs):
    total_batch_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        y_pred = model(images)
        loss = criterion(y_pred, labels)
        total_batch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_batch_loss = total_batch_loss / len(train_loader)
    training_loss[epoch] = avg_batch_loss
    print(f"Loss for epoch {epoch}: {avg_batch_loss:.4F}")


# Testing
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    total_test_loss = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()

        # Get predicted class
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
print("=================================================")
print(f"Average Test Loss: {avg_test_loss:.2F}")
print(classification_report(all_labels, all_preds))
print("=================================================")

# TODO
'''
# Improve accuracy (Current Accuracy = 47% (20 epochs); Target Accuracy = 60%)

- Do Data Augmentation 
- Use BatchNorm everywhere
- Play with layers and filters
- Train longer
'''

