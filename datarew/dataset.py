import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import torch
import torchvision
from torchvision import transforms
import numpy as np


def collate_fn(batch):
    return jnp.asarray(torch.stack([b[0] for b in batch], dim=0).permute(0, 2, 3, 1).cpu().numpy()), \
    jnp.array([b[1] for b in batch], dtype=jnp.int32)


def get_dataloaders_fmnist(corruption: float, batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    np.random.seed(0)

    trainset = torch.utils.data.Subset(torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform,
                                                target_transform=lambda y: np.random.randint(10) \
                                                if np.random.rand() < corruption else y), torch.arange(1000))  # attention

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0, collate_fn=collate_fn,
                                            drop_last=True)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0, collate_fn=collate_fn)

    # for outer optimization
    valloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=0, collate_fn=collate_fn)
    return trainloader, valloader, testloader


def get_dataloaders_cifar(corruption: float, batch_size: int):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    class ToNumpy:
        def __call__(self, pic):
            return np.asarray(pic.permute(1, 2, 0), dtype=np.float32)

    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ToNumpy(),
    ])

    np.random.seed(0)
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform,
                                              target_transform=lambda y: np.random.randint(10) \
                                                if np.random.rand() < corruption else y)

    num_train = len(train_data)
    split = int(np.floor(0.5 * num_train))

    trainloader = jdl.DataLoader(torch.utils.data.Subset(train_data, range(split)), 'pytorch',
                                batch_size=batch_size, shuffle=True, drop_last=True)

    valloader = jdl.DataLoader(torch.utils.data.Subset(train_data, range(split, len(train_data))), 'pytorch',
                                batch_size=batch_size, shuffle=True, drop_last=True)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
    testloader = jdl.DataLoader(test_data, 'pytorch',
                                batch_size=batch_size, shuffle=False, drop_last=False)

    return trainloader, valloader, testloader

