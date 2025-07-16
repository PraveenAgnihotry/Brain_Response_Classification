import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

def load_and_predict(directory, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # load dataset from the provided directory
    dataset = datasets.ImageFolder(root=directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # load model and weights
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    # predict
    labels = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).int().tolist()

            if isinstance(preds, int):
                preds = [preds]
            labels.extend(preds)

    return labels  # list of 0s and 1s
