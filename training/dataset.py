import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# -----------------------------
# Retina Preprocessing Function
# -----------------------------
def preprocess_retina(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate retina
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        image = image[y:y+h, x:x+w]

    # CLAHE Enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


# -----------------------------
# Dataset Class
# -----------------------------
class FundusDataset(Dataset):

    def __init__(self, dataframe, image_dir, train=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.train = train

        if train:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5
                ),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_name = self.df.iloc[idx]["image"]
        label = self.df.iloc[idx]["label"]

        img_path = f"{self.image_dir}/{img_name}"

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🔥 Apply Retina Preprocessing
        image = preprocess_retina(image)

        # Apply Augmentations
        image = self.transform(image=image)["image"]

        return image, torch.tensor(label)