import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np


class DataCleaningDS(Dataset):
    def __init__(self, ds):
        self.ds = ds
    
    def __getitem__(self, idx):
        return *self.ds[idx], idx
    
    def __len__(self):
        return len(self.ds)


def get_dataloaders_cifar(corruption: float, batch_size: int, num_samples, ds_name='cifar10'):
    CIFAR_10_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_10_STD = [0.24703233, 0.24348505, 0.26158768]
    CIFAR_100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR_100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    CIFAR_MEAN, CIFAR_STD = CIFAR_10_MEAN, CIFAR_10_STD
    if ds_name.endswith('100'):
        CIFAR_MEAN, CIFAR_STD = CIFAR_100_MEAN, CIFAR_100_STD

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
    ds_cls = torchvision.datasets.CIFAR10 if ds_name.endswith('10') else torchvision.datasets.CIFAR100
    train_data = ds_cls(root='./data', train=True, download=True, transform=train_transform,
                                              target_transform=lambda y: np.random.randint(10) \
                                                if np.random.rand() < corruption else y)
    ids = np.random.choice(len(train_data), size=(min(num_samples, len(train_data)),), replace=False)
    train_data = DataCleaningDS([train_data[i] for i in ids])
    trainloader = jdl.DataLoader(train_data, 'pytorch',
                                batch_size=batch_size, shuffle=True, drop_last=True)

    test_data = ds_cls(root='./data', train=False, download=True, transform=train_transform)
    testloader = jdl.DataLoader(test_data, 'pytorch',
                                batch_size=batch_size, shuffle=False, drop_last=False)
    valloader = jdl.DataLoader(test_data, 'pytorch',
                                batch_size=batch_size, shuffle=True, drop_last=True)
    return trainloader, valloader, testloader

