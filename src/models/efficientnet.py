import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights

def get_efficientnet_b4():

    model=models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    num_features=model.classifier[1].in_features
    model.classifier=nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features=num_features, out_features=1, bias=True))
    return model