import torch
import torchvision
from torchvision import transforms

def load_cifar(p=500, resize=None, train=False, device='cpu', class_=None):
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if resize is not None:
        test_list.append(transforms.Resize((resize, resize), interpolation=3))

    transform_test = transforms.Compose(test_list)

    testset = torchvision.datasets.CIFAR10(
        root='/home/lpetrini/data/cifar10', train=train, download=True, transform=transform_test)
    if class_ is not None:
        testset = torch.utils.data.Subset(testset, [i for i, t in enumerate(testset.targets) if t == class_])
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=True, num_workers=2)

    imgs, y = next(iter(testloader))
    return imgs.to(device), y.to(device)