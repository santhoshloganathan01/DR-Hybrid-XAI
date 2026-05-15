import torch
import torch.nn as nn
import timm


class HybridEffNetViT(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        # EfficientNetV2 backbone
        self.cnn = timm.create_model(
            "tf_efficientnetv2_m",
            pretrained=True,
            num_classes=0
        )

        # Vision Transformer
        self.vit = timm.create_model(
            "vit_base_patch16_384",
            pretrained=True,
            num_classes=0
        )

        cnn_features = self.cnn.num_features
        vit_features = self.vit.num_features

        self.classifier = nn.Sequential(
            nn.Linear(cnn_features + vit_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):

        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)

        fused = torch.cat((cnn_feat, vit_feat), dim=1)

        return self.classifier(fused)