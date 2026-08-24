import torch.nn as nn

from EEGModules import ConstraintDepthwiseConv2d, ConstraintLazyLinear, SeparableConv2d


class EEGNet(nn.Module):
    def __init__(self, num_channels=22, num_classes=4, dropout_rate=0.25):
        super().__init__()
        self.f1 = 8
        self.D = 2
        self.f2 = 16
        self.net = nn.Sequential(
            nn.Conv2d(1, self.f1, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(self.f1),
            ConstraintDepthwiseConv2d(self.f1, self.f1 * self.D, kernel_size=(num_channels, 1), max_norm=1),
            nn.BatchNorm2d(self.f1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropout_rate),
            SeparableConv2d(self.f1 * self.D, self.f2, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(self.f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout_rate),
            nn.Flatten(),
            ConstraintLazyLinear(num_classes, max_norm=0.25),
        )

    def forward(self, x):
        return self.net(x)


class EEGNetBackbone(nn.Module):
    def __init__(self, num_channels=22, num_classes=4, dropout_rate=0.25):
        super().__init__()
        self.f1 = 8
        self.D = 2
        self.f2 = 16
        self.net = nn.Sequential(
            nn.Conv2d(1, self.f1, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(self.f1),
            ConstraintDepthwiseConv2d(self.f1, self.f1 * self.D, kernel_size=(num_channels, 1), max_norm=1),
            nn.BatchNorm2d(self.f1 * self.D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropout_rate),
            SeparableConv2d(self.f1 * self.D, self.f2, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(self.f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout_rate),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)
