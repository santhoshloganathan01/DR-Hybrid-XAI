import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from training.dataset import FundusDataset
from training.loss import FocalLoss
from models.hybrid_model import HybridEffNetViT

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/labels.csv")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_number = 1
best_overall_acc = 0

# -----------------------------
# K-FOLD TRAINING
# -----------------------------
for train_idx, val_idx in skf.split(df, df["label"]):

    print(f"\n========== Fold {fold_number} ==========")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = FundusDataset(train_df, "data/images", train=True)
    val_dataset = FundusDataset(val_df, "data/images", train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    # -----------------------------
    # MODEL
    # -----------------------------
    model = HybridEffNetViT(num_classes=5).to(device)

    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20
    )

    scaler = torch.amp.GradScaler("cuda")

    best_fold_acc = 0

    # -----------------------------
    # EPOCH LOOP
    # -----------------------------
    for epoch in range(20):

        model.train()
        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        acc = accuracy_score(val_true, val_preds)
        f1 = f1_score(val_true, val_preds, average="weighted")

        print(
            f"Fold {fold_number} | "
            f"Epoch {epoch+1} | "
            f"Loss: {running_loss/len(train_loader):.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f}"
        )

        # Save best fold model
        if acc > best_fold_acc:
            best_fold_acc = acc
            torch.save(model.state_dict(), f"best_fold_{fold_number}.pth")

    print(f"Best Accuracy Fold {fold_number}: {best_fold_acc:.4f}")

    # Track best overall model
    if best_fold_acc > best_overall_acc:
        best_overall_acc = best_fold_acc
        torch.save(model, "full_hybrid_model.pth")
        print("Full best model saved!")

    fold_number += 1

print("\nTraining Complete.")
print("Best Overall Accuracy:", best_overall_acc)