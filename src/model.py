import torch.nn as nn
import math
import torchvision.models as models

class ShipModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ShipModel, self).__init__()

        self.model = models.resnet18(pretrained= False, num_classes = num_classes)
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=8, stride=2),
        # )
        # # self._require_grad_false()
        #
        # self.classifier = nn.Sequential(
        #     nn.Linear(384, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, num_classes),
        # )
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return x;

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _require_grad_false(self):
        for p in self.features.parameters():
            p.requires_grad = False
