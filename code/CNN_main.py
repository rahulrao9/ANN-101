
# to run use - python3 CNN_test.py --CNN_main

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Dataset Preparation
def prepareData(classes_list, class_dict):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Classes to include

    class_indices_to_keep = [0, 1, 2]

    # Training Dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    for i in range(len(train_dataset)):
        if (train_dataset.targets[i] in classes_list):
            train_dataset.targets[i] = class_dict[train_dataset.targets[i]]
        else:
            train_dataset.targets[i] = -1
    train_dataset = torch.utils.data.Subset(train_dataset, [i for i in range(len(train_dataset)) if
                                                            train_dataset.targets[i] in class_indices_to_keep])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Testing Dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    for i in range(len(test_dataset)):
        if (test_dataset.targets[i] in classes_list):
            test_dataset.targets[i] = class_dict[test_dataset.targets[i]]
        else:
            test_dataset.targets[i] = -1
    test_dataset = torch.utils.data.Subset(test_dataset, [i for i in range(len(test_dataset)) if
                                                          test_dataset.targets[i] in class_indices_to_keep])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader, test_loader


# The CNN Model Class
class CNN(nn.Module):

    # Define Initialization function
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.criterion = None

        # ----------------------------------------------

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 10)
        # ----------------------------------------------
        self.setCriterionAndOptimizer()

    # Define Forward Pass
    def forward(self, x):

        x = self.act1(self.conv1(x))

        x = self.pool2(x)

        x = self.flat(x)

        x = self.act3(self.fc3(x))
        x = self.drop3(x)

        x = self.fc4(x)

        return x

    # Set Values of self.optimizer and self.criterion
    def setCriterionAndOptimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)
        self.criterion = nn.CrossEntropyLoss()



# Input: 1) model: CNN object
# Output: 1) train_accuracy: float
def train(model, train_loader):

    loss_function = model.criterion
    optimizer = model.optimizer

    epochs = 3
    accuracy = 0
    for epoch in range(epochs):
        acc = 0
        count = 0
        for inputs, labels in train_loader:

            label_out = model(inputs)
            loss = loss_function(label_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy += (torch.argmax(label_out, 1) == labels).float().sum()
            count += len(labels)

        acc /= count
        accuracy = accuracy * 100

    return accuracy


# Implement evaluation here
# Input: 1) model: CNN object
# Output: 1) test_accuracy: float
def evaluate(model, test_loader):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    criterion = model.criterion

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate validation/test loss and accuracy
    test_loss = running_loss / len(test_loader)
    test_accuracy = 100.0 * correct_predictions / total_samples

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    return test_accuracy
