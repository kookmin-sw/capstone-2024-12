import torch
import torch.nn as nn

class ModelClass(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelClass, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = self._make_layer(64, 128, kernel_size=3, num_blocks=2, stride=1)
        self.layer3 = self._make_layer(128, 256, kernel_size=3, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, kernel_size=3, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, kernel_size, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # Update in_channels for the next block
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # 모델 인스턴스 생성 및 추론 준비
    model = ModelClass()
    model.eval()

    # 랜덤 입력 생성
    input_tensor = torch.randn(1, 3, 224, 224)

    # 추론 실행
    with torch.no_grad():
        output = model(input_tensor)

    print(output)

    torch.save(model.state_dict(), "torch.pt")
