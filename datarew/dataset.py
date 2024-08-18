import jax
import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np


def collate_fn(batch):
    return jnp.asarray(torch.stack([b[0] for b in batch], dim=0).permute(0, 2, 3, 1).cpu().numpy()), \
    jnp.array([b[1] for b in batch], dtype=jnp.int32)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


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


def get_dataloaders_cifar(batch_size: int, num_samples: int,
                          imbalance_factor=1, ds_name='cifar10'):
    name_to_cls = {'cifar10': torchvision.datasets.CIFAR10,
                   'cifar100': torchvision.datasets.CIFAR100,
                   'svhn': torchvision.datasets.SVHN,
                   'fmnist': torchvision.datasets.FashionMNIST,
                   'mnist': torchvision.datasets.MNIST
                   }
    name_to_mean = {'cifar10': (0.49139968, 0.48215827, 0.44653124),
                    'cifar100': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    'svhn': (0.4914, 0.4822, 0.4465),
                    'fmnist': (0.5,),
                    'mnist': (0.5,),
                    }
    name_to_std = {'cifar10': (0.24703233, 0.24348505, 0.26158768),
                   'cifar100': (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                   'svhn': (0.2023, 0.1994, 0.2010),
                   'fmnist': (0.5,),
                   'mnist': (0.5,)
                   }

    class ToNumpy:
        def __call__(self, pic):
            return np.asarray(pic.permute(1, 2, 0), dtype=np.float32)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(name_to_mean[ds_name], name_to_std[ds_name]),
        ToNumpy(),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(name_to_mean[ds_name], name_to_std[ds_name]),
        ToNumpy(),
    ])

    np.random.seed(0)
    ds_cls = name_to_cls[ds_name]
    try:
        train_data = ds_cls(root='./data', train=True, download=True, transform=train_transform)
    except:
        train_data = ds_cls(root='./data', split='train', download=True, transform=train_transform)

    ids = np.random.choice(len(train_data), size=(min(num_samples, len(train_data)),), replace=False)
    train_data = [train_data[i] for i in ids]
    num_train = len(train_data)
    split = int(np.floor(num_train * 0.8))

    train_ds = torch.utils.data.Subset(train_data, range(split))
    n_cls = 100 if ds_name == 'cifar100' else 10
    cls_to_images = {i: [] for i in range(n_cls)}
    for x, y in train_ds:
        cls_to_images[y].append(x)
    cls_to_ratio = {i: 1 / (imbalance_factor ** (i / (n_cls - 1))) for i in range(n_cls)}
    cls_to_images = {i : cls_to_images[i][:int(len(cls_to_images[i]) * cls_to_ratio[i])] for i in cls_to_images}
    train_ds = [(x, y) for y in cls_to_images for x in cls_to_images[y]]
    print('Trian ds len', len(train_ds))
    trainloader = DataLoader(train_ds, batch_size=batch_size,
                             shuffle=True, drop_last=True, collate_fn=numpy_collate)

    valloader = DataLoader(torch.utils.data.Subset(train_data, range(split, len(train_data))),
                           batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate)
    try:
        test_data = ds_cls(root='./data', train=False, download=True, transform=test_transforms)
    except:
        test_data = ds_cls(root='./data', split='test', download=True, transform=test_transforms)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False,
                            collate_fn=numpy_collate)

    return trainloader, valloader, testloader

