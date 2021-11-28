import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import timm
from torchvision import models


class ReadTimmModule(BaseModel):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = timm.create_model(
            model_name = self.model_name, num_classes=self.num_classes, pretrained = pretrained
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def init_weights(self,):
        if self.pretrained:
            self.model = timm.create_model(
            model_name = self.model_name, num_classes=self.num_classes, pretrained = self.pretrained 
        )


class ReadTorchvisionModule(BaseModel):  #
    def __init__(self, model_name, pretrained=True, classifier=None):
        super().__init__()

        self.model = eval(f"models.{model_name}(pretrained={pretrained})")
        self.model.fc = eval(classifier)

    def forward(self, x):
        x = self.model(x)
        return x
