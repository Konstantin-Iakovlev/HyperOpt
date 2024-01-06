import jax
import jax.numpy as jnp
import torch
import torchvision
from torchvision import transforms
import numpy as np


def collate_fn(batch):
    return jnp.asarray(torch.stack([b[0] for b in batch], dim=0).permute(0, 2, 3, 1).cpu().numpy()), \
    jnp.array([b[1] for b in batch], dtype=jnp.int32)


def get_dataloaders(corruption: float, batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    np.random.seed(0)

    trainset = torch.utils.data.Subset(torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform,
                                                target_transform=lambda y: torch.randint(0, 10, (1,)).item() \
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
