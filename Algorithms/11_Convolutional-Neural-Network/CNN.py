import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score

'''
# Find mean and standard deviation of training dataset to normalize the input for train as well as test dataset
# Note: Calculating mean and std only on train, because if we include test dataset also in this process, then it will be
#       considered as data leakage, because model will be trained on the data which is in test dataset. And will eventually
#       model performance.
# Define data transform (no normalization yet)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset_full = CIFAR100(root='F:\\Tmp\\Machine-Learning\\Datasets', train=True, download=True, transform=transform)
train_split_size = int(len(train_dataset_full)*0.8)
val_split_size = len(train_dataset_full) - train_split_size
train_data, val_data = torch.utils.data.random_split(train_dataset_full, [train_split_size, val_split_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False)
mean = 0.
std = 0.
total_images = 0
for images, _ in train_loader:
    batch_size = images.size(0)
    images = images.view(batch_size, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images += batch_size
mean /= total_images
std /= total_images
print(f'Mean: {mean}, Std: {std}')
# Output: Mean: tensor([0.5064, 0.4863, 0.4408]), Std: tensor([0.2008, 0.1983, 0.2022])
'''





# Mean and Std for CIFAR-100 Dataset (Use it for input normalization)
mean = [0.5064, 0.4863, 0.4408]
std = [0.2008, 0.1983, 0.2022]

# Define transformations for train and test datasets
# Apply Geometry and Colour augmentation on training dataset on the fly
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomCrop(32, padding=4, pad_if_needed=True),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.RandomRotation(degrees=15),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])
# No data augmentation for test dataset because we cannot evaluate model performance reliably, if we augment the training dataset
# test dataset is not natural anymore
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std)])

# Load training and testing dataset
train_data_full = CIFAR100(root="F:\\Tmp\\Machine-Learning\\Datasets", train=True, download=True, transform=train_transform)
test_data = CIFAR100(root="F:\\Tmp\\Machine-Learning\\Datasets", train=False, download=True, transform=test_transform)

# Train-Validation split
train_split_size = int(len(train_data_full)*0.8)
val_split_size = len(train_data_full) - train_split_size
train_data, val_data = torch.utils.data.random_split(train_data_full, [train_split_size, val_split_size])

'''
Training Dataset size: 40000
Validation Dataset size: 10000
Testing Dataset Size: 10000
'''
# Load data in batches
# 1024 is the optimal batch size for my current GPU (T1000)
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

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
            # Block 1/2/3: Two con_layer before pooling learn richer spatial features
            # Rule of thumb: Double channels after each pooling (why? richer features), Reduce spatial size, increase depth

            # Block 1: [Input size - 32x32x3] --> [Output size - 16x16x3]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # output depth = 64
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True), # inplace (default=False), it changes the output tensor directly, without creating new tensor --> memory efficient
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pooling reduce spatial size (original_size/2)

            # Block 2: [Input size - 16x16x3] ---> [Output Size - 8x8x3]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: [Input size - 8x8x3] ---> [Output Size - 4x4x3]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # Neural Network
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # PyTorch chooses the kernel_size and stride value to output 1x1 feature map
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=100),
        )

    def forward(self, x):
        conv_output = self.conv(x)
        y_pred = self.linear(conv_output)
        return y_pred

# Define model, loss function, number of epochs & optimizer
model = ConvolutionalNet()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# Store training and validation loss
training_loss = []
validation_loss = []

# Early stopping stats
early_stopping_obj = EarlyStopping.EarlyStopping(10)

# Training
for epoch in range(epochs):
    # Train
    model.train()
    total_train_batch_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        total_train_batch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_batch_loss = total_train_batch_loss / len(train_loader)
    training_loss.append(avg_train_batch_loss)
    print("=================================================")
    print(f"Training loss for epoch {epoch}: {avg_train_batch_loss:.4F}")

    # Validate
    model.eval()
    with torch.no_grad():
        total_val_batch_loss = 0.0
        all_pred = []
        all_labels = []
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, preds = torch.max(output, dim=1)
            all_pred.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = criterion(output, labels)
            total_val_batch_loss += loss.item()
        avg_val_batch_loss = total_val_batch_loss / len(val_loader)
        validation_loss.append(avg_val_batch_loss)
        accuracy = (accuracy_score(all_pred, all_labels) * 100)
        print(f"Validation loss for epoch {epoch}: {avg_val_batch_loss:.4f}")
        print(f"Accuracy for epoch {epoch}: {accuracy:.2f}%")
        early_stopping_obj(avg_val_batch_loss)
        if early_stopping_obj.early_stop:
            # Save weights of best model
            print(f"============== Early Stopping triggered after {epoch} epochs! ====================")
            break


# Plot training and validation loss
epochs_range = range(len(training_loss))
plt.plot(epochs_range, training_loss, color='red')
plt.plot(epochs_range, validation_loss, color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()


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
print("=================================================")
print(classification_report(all_labels, all_preds))
print("=================================================")


