import torch
import torch.nn as nn
from torchvision import models

class GoatDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(GoatDetector, self).__init__()
        self.resnet = models.resnet18(pretrained)
        self.final_layer = nn.Linear(1000, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.final_layer(self.resnet(x)))

# discovered by threshold searching
threshold = 0.76