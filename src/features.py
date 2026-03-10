import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    Extract intermediate feature maps from a selectable ResNet-family backbone.
    We return layer2 and layer3 maps (common PatchCore choice).
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        elif backbone == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Returns:
            f2: feature map from layer2
            f3: feature map from layer3
        """
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3
