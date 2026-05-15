import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# CLASS LABELS
############################################

classes = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

############################################
# MODEL
############################################

class HybridEffNetViT(nn.Module):

    def __init__(self,num_classes=5):

        super().__init__()

        self.cnn = timm.create_model(
            "tf_efficientnetv2_m",
            pretrained=False,
            num_classes=0
        )

        self.vit = timm.create_model(
            "vit_base_patch16_384",
            pretrained=False,
            num_classes=0,
            img_size=384
        )

        cnn_features = self.cnn.num_features
        vit_features = self.vit.num_features

        self.classifier = nn.Sequential(

            nn.Linear(cnn_features+vit_features,768),

            nn.BatchNorm1d(768),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(768,num_classes)
        )

    def forward(self,x):

        cnn_feat = self.cnn(x)

        vit_feat = self.vit(x)

        fused = torch.cat((cnn_feat,vit_feat),dim=1)

        return self.classifier(fused)

############################################
# LOAD MODEL
############################################

model = HybridEffNetViT()

model.load_state_dict(
    torch.load("best_hybrid_model.pth",map_location=device)
)

model.to(device)

model.eval()

print("Model loaded successfully")


############################################
# PREPROCESS IMAGE
############################################

def preprocess(image_path):

    img = cv2.imread(image_path)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    img = cv2.cvtColor(merged,cv2.COLOR_LAB2BGR)

    img = cv2.resize(img,(384,384))

    img = img/255.0

    img = np.transpose(img,(2,0,1))

    tensor = torch.tensor(img,dtype=torch.float32)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    return tensor,img


############################################
# PREDICT
############################################

def predict(image_path):

    tensor,img = preprocess(image_path)

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.softmax(outputs,dim=1)

        confidence,pred = torch.max(probs,1)

    pred_class = classes[pred.item()]

    conf = confidence.item()

    print("\nPrediction:",pred_class)

    print("Confidence:",round(conf,3))

    print("\nClass Probabilities")

    for i,p in enumerate(probs[0]):

        print(classes[i],":",round(p.item(),3))

    return img


############################################
# RUN
############################################

image_path = input("Enter image path: ")

img = predict(image_path)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis("off")
plt.show()