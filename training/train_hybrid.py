import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataset import FundusDataset
from models.hybrid_model import HybridEffNetViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = FundusDataset(
    csv_file="data/labels.csv",
    img_dir="data/images"
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = HybridEffNetViT(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()

for epoch in range(2):  # quick test
    print(f"\nEpoch {epoch+1}")

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 30 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

print("Hybrid training complete.")
