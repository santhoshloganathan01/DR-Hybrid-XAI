import cv2
import torch
from training.preprocessing import preprocess

img = cv2.imread("data/images/000c1434d8d7.png")

processed = preprocess(img)

tensor = torch.tensor(processed).permute(2,0,1).float()

print("Tensor shape:", tensor.shape)
