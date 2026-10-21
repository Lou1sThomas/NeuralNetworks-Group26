import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, blocks_per_stage=(2,2,2,2), widths=(64, 128, 256, 512), dropout=0.0):
        super().__init__()

        # first conv layer
        self.stem = nn.Sequential(nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True),
                                  )

        stages = []
        current_channels = widths[0]
        for i, (n_blocks, width) in enumerate(zip(blocks_per_stage, widths)):
            stride = 1 if i == 0 else 2
            stages.append(self._make_stage(current_channels, width, n_blocks, stride))
            current_channels = width
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d((1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(current_channels, num_classes)
        self._init_weights()

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
            layers = [ResidualBlock(in_channels, out_channels, stride)]
            for _ in range(1, n_blocks):
                layers.append(ResidualBlock(out_channels, out_channels, stride=1))
                return nn.Sequential(*layers)

    def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
            x = self.stem(x)
            x = self.stages(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

if __name__ == "__main__":
    NUM_CLASSES = 10
    IMG_SIZE = 64

    model = ResNet(num_classes = NUM_CLASSES)
    x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    out = model(x)
    print("Output shape:", out.shape)  

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")