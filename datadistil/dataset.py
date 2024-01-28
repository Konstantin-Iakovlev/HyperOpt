import jax
import jax.numpy as jnp
from torch.utils.data import Dataset, DataLoader
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

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_dataloaders(batch_size, ds_name='cifar10'):
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
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
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

    np.random.seed(0)
    ds_cls = name_to_cls[ds_name]
    train_data = ds_cls(root='./data', train=True, download=True, transform=train_transform)
    ids = np.random.choice(len(train_data), size=(len(train_data),), replace=False)
    train_data = DataCleaningDS([train_data[i] for i in ids])
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate)

    test_data = ds_cls(root='./data', train=False, download=True, transform=train_transform)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=numpy_collate)
    valloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate)
    return trainloader, valloader, testloader

