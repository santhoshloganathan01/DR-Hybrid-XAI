import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from torch.utils.data import Dataset, DataLoader

from models.hybrid_model import HybridEffNetViT


# ------------------------
# Dataset Loader
# ------------------------

class FundusDataset(Dataset):

    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = f"{self.img_dir}/{row.image}"

        image = cv2.imread(img_path)

        image = cv2.resize(image, (384,384))

        image = image / 255.0

        image = np.transpose(image,(2,0,1))

        image = torch.tensor(image,dtype=torch.float32)

        label = int(row.label)

        return image,label


# ------------------------
# Load Model
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridEffNetViT(num_classes=5)

model.load_state_dict(
    torch.load("best_hybrid_model.pth", map_location=device)
)

model.to(device)
model.eval()


# ------------------------
# Load Dataset
# ------------------------

df = pd.read_csv("data/labels.csv")

dataset = FundusDataset(df,"data/images")

loader = DataLoader(dataset,batch_size=8,shuffle=False)


# ------------------------
# Collect predictions
# ------------------------

y_true = []
y_score = []

with torch.no_grad():

    for images,labels in loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs,dim=1)

        y_score.extend(probs.cpu().numpy())

        y_true.extend(labels.numpy())


y_true = np.array(y_true)
y_score = np.array(y_score)


# ------------------------
# Convert labels to one-hot
# ------------------------

y_true_bin = label_binarize(y_true, classes=[0,1,2,3,4])

n_classes = 5


# ------------------------
# ROC Calculation
# ------------------------

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(
        y_true_bin[:, i],
        y_score[:, i]
    )

    roc_auc[i] = auc(fpr[i], tpr[i])


# ------------------------
# Plot ROC
# ------------------------

plt.figure(figsize=(8,6))

for i in range(n_classes):

    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {i} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Diabetic Retinopathy Detection")

plt.legend(loc="lower right")

plt.show()