import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),  # scales to [0,1]
])

from scipy.ndimage import gaussian_filter, map_coordinates
import random

def elastic_deformation(image, alpha=36, sigma=6):
    image = image.squeeze().numpy()  # (1, 28, 28) → (28, 28)
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return torch.tensor(distorted).unsqueeze(0)

# Load dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset: 80% train, 20% test
train_size = int(0.8 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size
train_dataset, test_dataset = random_split(mnist_dataset, [train_size, test_size])

# Optional: Apply elastic deformation to samples (example)
def augment_with_elastic(dataset, num_samples=1000):
    augmented_images = []
    augmented_labels = []

    for i in range(num_samples):
        img, label = dataset[i]
        deformed = elastic_deformation(img)
        augmented_images.append((deformed, label))

    return augmented_images

# Example usage
augmented_samples = augment_with_elastic(train_dataset, num_samples=1000)

# Show a sample
fig, axs = plt.subplots(1, 2)
sample_img, _ = train_dataset[0]
deformed_img = elastic_deformation(sample_img)

axs[0].imshow(sample_img.squeeze(), cmap='gray')
axs[0].set_title("Original")
axs[1].imshow(deformed_img.squeeze(), cmap='gray')
axs[1].set_title("Elastic Deformed")
plt.show()



from collections import defaultdict
from torchvision.datasets import MNIST

def create_classwise_dict(dataset):
    classwise_data = defaultdict(list)
    for img, label in dataset:
        classwise_data[label].append(img)
    return classwise_data

import random

def create_episode(classwise_data, N=5, K=1, Q=1):
    """
    Create a single N-way K-shot episode with Q query samples per class.
    """
    selected_classes = random.sample(list(classwise_data.keys()), N)

    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    label_map = {cls: idx for idx, cls in enumerate(selected_classes)}

    for cls in selected_classes:
        samples = random.sample(classwise_data[cls], K + Q)
        support = samples[:K]
        query = samples[K:]

        support_images.extend(support)
        support_labels.extend([label_map[cls]] * K)
        query_images.extend(query)
        query_labels.extend([label_map[cls]] * Q)

    return (support_images, support_labels), (query_images, query_labels)

def compute_accuracy(preds, labels):
    correct = sum([p == l for p, l in zip(preds, labels)])
    return correct / len(labels)

# Step 1: Organize training data by class
classwise_train_data = create_classwise_dict(train_dataset)

# Step 2: Generate an episode
(support_imgs, support_lbls), (query_imgs, query_lbls) = create_episode(classwise_train_data, N=5, K=1, Q=3)

# Step 3: Visualize a few support and query images
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    axs[0][i].imshow(support_imgs[i].squeeze(), cmap='gray')
    axs[0][i].set_title(f"Support: {support_lbls[i]}")
    axs[1][i].imshow(query_imgs[i].squeeze(), cmap='gray')
    axs[1][i].set_title(f"Query: {query_lbls[i]}")
    axs[0][i].axis('off')
    axs[1][i].axis('off')
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

# Common Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --------------------------------------------
# 3.1 Prototypical Network
# --------------------------------------------
class ProtoNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)

def compute_prototypes(embeddings, labels, N):
    prototypes = []
    for i in range(N):
        cls_embeds = embeddings[labels == i]
        prototype = cls_embeds.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

def euclidean_distance(a, b):
    return torch.cdist(a, b, p=2)

def create_episode(classwise_data, N, K, Q):
    selected_classes = random.sample(list(classwise_data.keys()), N)
    support_imgs, query_imgs = [], []
    support_lbls, query_lbls = [], []
    for i, cls in enumerate(selected_classes):
        imgs = random.sample(classwise_data[cls], K + Q)
        support_imgs += imgs[:K]
        query_imgs += imgs[K:]
        support_lbls += [i]*K
        query_lbls += [i]*Q
    return (support_imgs, support_lbls), (query_imgs, query_lbls)

def train_protonet(model, classwise_data, episodes=1000, N=5, K=1, Q=5, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for episode in range(episodes):
        (support_imgs, support_lbls), (query_imgs, query_lbls) = create_episode(classwise_data, N, K, Q)
        support = torch.stack([transform(img) for img in support_imgs]).to(device)
        query = torch.stack([transform(img) for img in query_imgs]).to(device)
        support_lbls = torch.tensor(support_lbls, dtype=torch.long).to(device)
        query_lbls = torch.tensor(query_lbls, dtype=torch.long).to(device)

        support_embed = model(support)
        query_embed = model(query)
        prototypes = compute_prototypes(support_embed, support_lbls, N)

        dists = euclidean_distance(query_embed, prototypes)
        logits = -dists  # More negative = more similar
        loss = loss_fn(logits, query_lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            preds = torch.argmin(dists, dim=1)
            acc = (preds == query_lbls).float().mean().item()
            print(f"[ProtoNet] Episode {episode+1}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

def evaluate_protonet(model, classwise_data, N=5, K=1, Q=5, episodes=100, device='cuda'):
    model.to(device)
    model.eval()
    accs = []

    with torch.no_grad():
        for _ in range(episodes):
            (support_imgs, support_lbls), (query_imgs, query_lbls) = create_episode(classwise_data, N, K, Q)
            support = torch.stack([transform(img) for img in support_imgs]).to(device)
            query = torch.stack([transform(img) for img in query_imgs]).to(device)
            support_lbls = torch.tensor(support_lbls).to(device)
            query_lbls = torch.tensor(query_lbls).to(device)

            support_embed = model(support)
            query_embed = model(query)
            prototypes = compute_prototypes(support_embed, support_lbls, N)
            dists = euclidean_distance(query_embed, prototypes)
            preds = torch.argmin(dists, dim=1)
            accs.append((preds == query_lbls).float().mean().item())

    print(f"[ProtoNet Eval] Avg Accuracy: {np.mean(accs):.4f}")

# --------------------------------------------
# 3.2 Siamese Network
# --------------------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)
        loss = 0.5 * (label * dist.pow(2) + (1 - label) * (F.relu(self.margin - dist).pow(2)))
        return loss.mean()

def make_pairs(data):
    pairs, labels = [], []
    label_dict = defaultdict(list)
    for img, lbl in data:
        label_dict[lbl].append(img)

    labels_list = list(label_dict.keys())
    for label in labels_list:
        imgs = label_dict[label]
        for i in range(len(imgs)-1):
            pairs.append([transform(imgs[i]), transform(imgs[i+1])])
            labels.append(1.0)
            neg_label = random.choice([l for l in labels_list if l != label])
            pairs.append([transform(imgs[i]), transform(label_dict[neg_label][0])])
            labels.append(0.0)
    return pairs, labels

class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __getitem__(self, index):
        x1, x2 = self.pairs[index]
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x1, x2, y

    def __len__(self):
        return len(self.pairs)

def train_siamese(model, loader, optimizer, criterion, device, epochs=5):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out1, out2 = model(x1, x2)
            loss = criterion(out1, out2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Siamese] Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

def evaluate_siamese(model, loader, device, threshold=0.5):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out1, out2 = model(x1, x2)
            dist = F.pairwise_distance(out1, out2)
            preds = (dist < threshold).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"[Siamese Eval] Accuracy: {correct / total:.4f}")
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import defaultdict
import random
import numpy as np

# Matching Network Encoder
class MatchingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# Cosine similarity classifier
def cosine_similarity_classifier(q_embed, s_embed, s_labels, N):
    sims = F.cosine_similarity(q_embed.unsqueeze(1), s_embed.unsqueeze(0), dim=2)
    weights = F.softmax(sims, dim=1)
    one_hot = F.one_hot(s_labels, num_classes=N).float()
    probs = torch.matmul(weights, one_hot)
    return torch.argmax(probs, dim=1)

# Create one-shot pairs (support + query)
def create_one_shot_task(data, N=5, K=1):
    classes = random.sample(list(data.keys()), N)
    support_images, support_labels = [], []
    query_images, query_labels = [], []
    for idx, cls in enumerate(classes):
        samples = random.sample(data[cls], K + 1)
        support_images.extend(samples[:K])
        support_labels.extend([idx] * K)
        query_images.append(samples[-1])
        query_labels.append(idx)
    return (support_images, support_labels), (query_images, query_labels)

# Training the Matching Network
def train_matching_net(model, data, epochs=300, N=5, K=1, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    transform = transforms.ToTensor()

    for ep in range(epochs):
        (s_imgs, s_lbls), (q_imgs, q_lbls) = create_one_shot_task(data, N, K)
        s = torch.stack([transform(img) for img in s_imgs]).to(device)
        q = torch.stack([transform(img) for img in q_imgs]).to(device)
        s_lbls = torch.tensor(s_lbls).to(device)
        q_lbls = torch.tensor(q_lbls).to(device)

        s_embed = model(s)
        q_embed = model(q)
        sims = F.cosine_similarity(q_embed.unsqueeze(1), s_embed.unsqueeze(0), dim=2)
        loss = loss_fn(sims, q_lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            preds = cosine_similarity_classifier(q_embed, s_embed, s_lbls, N)
            acc = (preds == q_lbls).float().mean().item()
            print(f"[{ep+1}/{epochs}] Loss: {loss.item():.4f}, Acc: {acc:.4f}")

# Dataset organization
def build_classwise_data():
    mnist = datasets.MNIST(root='./data', train=True, download=True)
    classwise_data = defaultdict(list)
    for img, label in mnist:
        classwise_data[label].append(img)
    return classwise_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classwise_data = build_classwise_data()
model = MatchingEncoder()
train_matching_net(model, classwise_data, epochs=300, N=5, K=1, device=device)





class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        return self.fc(x.view(x.size(0), -1))

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return F.pairwise_distance(out1, out2)

# Create pairs with labels (same class => 1, different => 0)
def create_siamese_pairs(data, num_pairs=1000):
    pairs = []
    labels = []
    transform = transforms.ToTensor()
    classes = list(data.keys())
    for _ in range(num_pairs):
        # Same class
        cls = random.choice(classes)
        img1, img2 = random.sample(data[cls], 2)
        pairs.append((transform(img1), transform(img2)))
        labels.append(1)

        # Different class
        cls1, cls2 = random.sample(classes, 2)
        img1 = random.choice(data[cls1])
        img2 = random.choice(data[cls2])
        pairs.append((transform(img1), transform(img2)))
        labels.append(0)
    return pairs, labels

# Train Siamese Model
def train_siamese(model, data, device='cuda', epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        pairs, labels = create_siamese_pairs(data)
        x1 = torch.stack([p[0] for p in pairs]).to(device)
        x2 = torch.stack([p[1] for p in pairs]).to(device)
        lbls = torch.tensor(labels, dtype=torch.float32).to(device)

        out = model(x1, x2)
        loss = loss_fn(-out, lbls)  # use -dist for BCE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[{epoch+1}/{epochs}] Siamese Loss: {loss.item():.4f}")

# Run training
siamese_model = SiameseNet()
train_siamese(siamese_model, classwise_data, device=device, epochs=10)




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

def print_metrics(name, acc, prec, rec, f1):
    print(f"\n--- {name} Performance ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

def evaluate_matching(model, data, N=5, K=1, num_tasks=100, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    transform = transforms.ToTensor()
    all_preds, all_labels = [], []

    for _ in range(num_tasks):
        (s_imgs, s_lbls), (q_imgs, q_lbls) = create_one_shot_task(data, N, K)
        s = torch.stack([transform(img) for img in s_imgs]).to(device)
        q = torch.stack([transform(img) for img in q_imgs]).to(device)
        s_lbls = torch.tensor(s_lbls).to(device)
        q_lbls = torch.tensor(q_lbls).to(device)

        s_embed = model(s)
        q_embed = model(q)
        preds = cosine_similarity_classifier(q_embed, s_embed, s_lbls, N)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(q_lbls.cpu().numpy())

    return evaluate_metrics(all_labels, all_preds)


def evaluate_siamese(model, classwise_data, device='cpu'):
    model.eval()
    num_pairs = 1000  # or 500, or however many you want
    pairs, labels = create_siamese_pairs(classwise_data, num_pairs=num_pairs)

    x1 = torch.stack([p[0] for p in pairs]).to(device)
    x2 = torch.stack([p[1] for p in pairs]).to(device)
    y_true = torch.tensor(labels).cpu().numpy()

    with torch.no_grad():
        distances = model(x1, x2).cpu()
    y_pred = (distances < 1.0).int().numpy()  # Threshold can be adjusted

    return evaluate_metrics(y_true, y_pred)

# Run evaluation for both models
matching_acc, matching_prec, matching_rec, matching_f1 = evaluate_matching(model, classwise_data, device=device)
print_metrics("Matching Network", matching_acc, matching_prec, matching_rec, matching_f1)

siamese_acc, siamese_prec, siamese_rec, siamese_f1 = evaluate_siamese(siamese_model, classwise_data, device=device)
print_metrics("Siamese Network", siamese_acc, siamese_prec, siamese_rec, siamese_f1)

def reduce_training_data(data, max_per_class=20):
    new_data = defaultdict(list)
    for cls in data:
        new_data[cls] = data[cls][:max_per_class]
    return new_data

# Compare performance with reduced data
reduced_data = reduce_training_data(classwise_data, max_per_class=10)
matching_acc_r, _, _, _ = evaluate_matching(model, reduced_data)
print(f"\nMatching Network with less data - Accuracy: {matching_acc_r:.4f}")

