import torch
import torch.nn as nn
from torchvision import models

class GoatDetector(nn.Module):
    def __init__(self):
        super(GoatDetector, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.final_layer = nn.Linear(1000, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.final_layer(self.resnet(x)))
