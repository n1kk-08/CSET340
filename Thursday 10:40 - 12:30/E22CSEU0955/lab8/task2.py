import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_datasets():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_resnet_model(version=18, num_classes=100):
    if version == 18:
        model = models.resnet18(pretrained=False)
    elif version == 34:
        model = models.resnet34(pretrained=False)
    else:
        raise ValueError("Unsupported ResNet version")

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def train_model(model, train_loader, test_loader, epochs=25, _='resnet'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    inference_times = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        test_loss, test_acc, inf_time = evaluate_model(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        inference_times.append(inf_time)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'Inference Time: {inf_time:.4f}s')

    stats = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'inference_times': inference_times
    }

    return stats

def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    inference_time = time.time() - start_time
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, inference_time

def plot_training_curves(stats1, stats2, label1='ResNet-18', label2='ResNet-34'):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(stats1['train_accs'], label=f'{label1} Train')
    plt.plot(stats1['test_accs'], label=f'{label1} Test')
    plt.plot(stats2['train_accs'], label=f'{label2} Train')
    plt.plot(stats2['test_accs'], label=f'{label2} Test')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(stats1['train_losses'], label=f'{label1} Train')
    plt.plot(stats1['test_losses'], label=f'{label1} Test')
    plt.plot(stats2['train_losses'], label=f'{label2} Train')
    plt.plot(stats2['test_losses'], label=f'{label2} Test')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def compare_models(stats1, stats2, model1, model2, name1='ResNet-18', name2='ResNet-34'):

    MODEL = 'Model'
    BEST_TRAIN_ACC = 'Best Train Acc (%)'
    BEST_TEST_ACC = 'Best Test Acc (%)'
    FINAL_TRAIN_LOSS = 'Final Train Loss'
    FINAL_TEST_LOSS = 'Final Test Loss'
    AVG_INFERENCE_TIME = 'Avg Inference Time (s)'
    PARAMETERS = 'Parameters'

    comparison = {
        MODEL: [name1, name2],
        BEST_TRAIN_ACC: [max(stats1['train_accs']), max(stats2['train_accs'])],
        BEST_TEST_ACC: [max(stats1['test_accs']), max(stats2['test_accs'])],
        FINAL_TRAIN_LOSS: [stats1['train_losses'][-1], stats2['train_losses'][-1]],
        FINAL_TEST_LOSS: [stats1['test_losses'][-1], stats2['test_losses'][-1]],
        AVG_INFERENCE_TIME: [np.mean(stats1['inference_times']), np.mean(stats2['inference_times'])],
        PARAMETERS: [sum(p.numel() for p in model1.parameters()), sum(p.numel() for p in model2.parameters())]
    }

    print("\nModel Comparison:")
    print("{:<15} {:<18} {:<18} {:<18} {:<18} {:<22} {:<15}".format(
        MODEL, BEST_TRAIN_ACC, BEST_TEST_ACC, FINAL_TRAIN_LOSS,
        FINAL_TEST_LOSS, AVG_INFERENCE_TIME, PARAMETERS))

    for i in range(2):
        print("{:<15} {:<18.2f} {:<18.2f} {:<18.4f} {:<18.4f} {:<22.4f} {:<15,}".format(
            comparison[MODEL][i],
            comparison[BEST_TRAIN_ACC][i],
            comparison[BEST_TEST_ACC][i],
            comparison[FINAL_TRAIN_LOSS][i],
            comparison[FINAL_TEST_LOSS][i],
            comparison[AVG_INFERENCE_TIME][i],
            comparison[PARAMETERS][i]))


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed()

    train_loader, test_loader = load_datasets()

    resnet18 = get_resnet_model(version=18).to(device)
    resnet34 = get_resnet_model(version=34).to(device)

    print("\nResNet-18 Summary:")
    summary(resnet18, (3, 32, 32))
    print("\nResNet-34 Summary:")
    summary(resnet34, (3, 32, 32))

    epochs = 5

    print("\nTraining ResNet-18...")
    resnet18_stats = train_model(resnet18, train_loader, test_loader, epochs, 'resnet18')

    print("\nTraining ResNet-34...")
    resnet34_stats = train_model(resnet34, train_loader, test_loader, epochs, 'resnet34')

    plot_training_curves(resnet18_stats, resnet34_stats)

    compare_models(resnet18_stats, resnet34_stats, resnet18, resnet34)