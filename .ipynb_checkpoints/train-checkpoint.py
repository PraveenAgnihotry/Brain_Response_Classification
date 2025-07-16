import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = r"D:\Study\Deep Learning\Exercise_1_v2\topomaps"
model_path = r"D:\Study\Deep Learning\Exercise_1_v2\model.pth"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset using ImageFolder (expects subdirectories as class names)
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Map "bad" to 0 and "good" to 1 explicitly if needed (already default behavior for alphabetical names)
class_to_idx = full_dataset.class_to_idx  # {'bad': 0, 'good': 1}
print(f"Class to index mapping: {class_to_idx}")

# Count class instances
targets = [label for _, label in full_dataset]
class_counts = torch.bincount(torch.tensor(targets))
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Train/validation split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create sample weights only for the train set
train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
class_counts = torch.bincount(torch.tensor(train_targets))
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[t] for t in train_targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model (you can replace with a better one like ResNet18)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
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

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Validation accuracy
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

# Save the trained model
# torch.save(model.state_dict(), model_path)
# print("Model saved to model.pth")
