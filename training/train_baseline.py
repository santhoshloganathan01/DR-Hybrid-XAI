import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataset import FundusDataset
from models.baseline_cnn import BaselineCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
dataset = FundusDataset(
    csv_file="data/labels.csv",
    img_dir="data/images"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = BaselineCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train 1 epoch
model.train()

for batch_idx, (images, labels) in enumerate(loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 50 == 0:
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

print("Training complete.")
