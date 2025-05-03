import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import os

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Clear cache before training
torch.cuda.empty_cache()

# -------------------------
# CIFAR-100 Classification with AlexNet & VGG16
# -------------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)  # Reduce batch size to avoid OOM
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)


# ------------------------
# AlexNet Model
# ------------------------
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)  # Fix weight loading
alexnet.classifier[6] = nn.Linear(4096, 100)
alexnet.to(device)

alexnet_criterion = nn.CrossEntropyLoss()
alexnet_optimizer = optim.SGD(alexnet.parameters(), lr=0.0001, momentum=0.9)  # Changed optimizer

train_model(alexnet, train_loader, alexnet_criterion, alexnet_optimizer)
evaluate_model(alexnet, test_loader)




