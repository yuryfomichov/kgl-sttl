import torch.nn as nn
from torch.nn import functional as F
import math
import torchvision.models as models

class ShipModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ShipModel, self).__init__()

        self.features = models.densenet121(pretrained = True, num_classes = 1000).features
        self.classifier = nn.Sequential(
            nn.Linear(1024,2048),
            nn.ReLU(True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, num_classes),
        )
        self._require_grad_false()
        self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=4).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _require_grad_false(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def densenet63(self, **kwargs):
        return DenseNet(num_init_features=64, growth_rate=32, block_config=(3, 4, 8, 6), num_classes=2)


class DenseNet(models.DenseNet):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__(growth_rate=growth_rate, block_config=block_config,
                 num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate, num_classes=num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=4).view(features.size(0), -1)
        out = self.classifier(out)
        return out


