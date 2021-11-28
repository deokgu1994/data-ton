import torch.nn as nn
import torch.optim as optim
from torchvision import models


class SetSegmentationTorchvision(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()

        self.model = eval(f"models.segmentation.{model_name}(pretrained={pretrained})")
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return x
