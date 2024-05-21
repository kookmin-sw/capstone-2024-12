import torch

def ModelClass():
    class BasicBlock(torch.nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(planes)
            self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(planes)

            self.shortcut = torch.nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(self.expansion * planes)
                )

        def forward(self, x):
            out = torch.nn.ReLU()(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.nn.ReLU()(out)
            return out

    class ResNet(torch.nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = torch.nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return torch.nn.Sequential(*layers)

        def forward(self, x):
            out = torch.nn.ReLU()(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = torch.nn.AdaptiveAvgPool2d((1, 1))(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    return ResNet(BasicBlock, [3, 4, 6, 3])
