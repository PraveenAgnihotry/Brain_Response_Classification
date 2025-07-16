import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split

# to remove randomness so the output can be reproduced
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 32
EPOCHS = 20

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for data and model
# data_dir = r"D:\Study\Deep Learning\Exercise_1_v2\topomaps"
# model_path = r"D:\Study\Deep Learning\Exercise_1_v2\model.pth"
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "topomaps")
model_path = os.path.join(base_dir, "model.pth")

# Image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])


# loading the dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

class_to_idx = full_dataset.class_to_idx  # {'bad': 0, 'good': 1}
print(f"Class to index mapping: {class_to_idx}")

# count class instances
targets = [label for _, label in full_dataset] # list of target labels
class_counts = torch.bincount(torch.tensor(targets))
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True) # because the classes are imbalanced

# train - validation split
train_size = int(0.80 * len(full_dataset)) # 85% for training because of the data is less
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create sample weights for train set
train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
class_counts = torch.bincount(torch.tensor(train_targets))
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[t] for t in train_targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# defining the ResNet 18 model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
model = model.to(device)    # move the model to the GPU

# defining the loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------- training loop --------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

    # validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# save the trained model
torch.save(model.state_dict(), model_path)
print("Model saved to model.pth")
