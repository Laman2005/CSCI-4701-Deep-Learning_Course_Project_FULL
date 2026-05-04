import torch.nn as nn
import torchvision.models as models

def get_model():

    model = models.efficientnet_b0(weights="DEFAULT")

    for p in model.features.parameters():
        p.requires_grad = False

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 1)
    )

    return model