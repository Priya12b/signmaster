import torch
import torch.nn as nn
from torchvision import models

def get_asl_model(num_classes=26, pretrained=True, architecture="mobilenet_v2"):
    """
    Builds a lightweight CNN model for ASL recognition.
    Supports MobileNetV2 (recommended for high accuracy & fast inference) and custom CNN.
    """
    if architecture == "mobilenet_v2":
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v2(weights=weights)
        except Exception:
            model = models.mobilenet_v2(pretrained=pretrained)
        
        # Replace the final classification head for 26 classes (A-Z)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        return model

    elif architecture == "custom_cnn":
        class ASLCustomCNN(nn.Module):
            def __init__(self, num_classes=26):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(256 * 4 * 4, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, num_classes)
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        return ASLCustomCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
