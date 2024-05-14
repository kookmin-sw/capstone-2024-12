import torchvision
import torchvision.transforms as transforms
import os


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cwd = os.getcwd()

# 학습 데이터셋 다운로드
trainset = torchvision.datasets.CIFAR10(root=f'{cwd}/example/ResNet34_CIFAR-10/', train=True,
                                        download=True, transform=transform)

# 테스트 데이터셋 다운로드
testset = torchvision.datasets.CIFAR10(root=f'{cwd}/example/ResNet34_CIFAR-10/', train=False,
                                       download=True, transform=transform)