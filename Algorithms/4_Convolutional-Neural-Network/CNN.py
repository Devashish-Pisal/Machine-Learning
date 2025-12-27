import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

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
        self.conv = nn.Sequential(
            # TODO

        )