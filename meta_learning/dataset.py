import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, dataset, n_way, n_shots, val_ratio=0.2, batch_size=128):
        self.n_way = n_way
        self.batch_size = batch_size
        num_cls_total = len(set([x[1] for x in dataset]))
        num_trn_cls = int(num_cls_total * (1 - val_ratio))
        num_val_cls = num_cls_total - num_trn_cls
        ids = np.random.choice(num_cls_total, num_trn_cls + num_val_cls, replace=False)
        self.trn_cls_to_obj = {d: [] for d in ids[:num_trn_cls]}
        self.test_cls_to_obj = {d: [] for d in ids[num_trn_cls:]}
        for x, y in dataset:
            if y in self.trn_cls_to_obj:
                self.trn_cls_to_obj[y].append(x)
            else:
                self.test_cls_to_obj[y].append(x)
        print('trn/val classes: ', len(self.trn_cls_to_obj), len(self.test_cls_to_obj))
        self.n_shots = n_shots
        self.drop_last = False if n_shots * n_way < batch_size else True

    def _prep_ds(self, ds):
        return {'image': np.stack([b[0] for b in ds], axis=0),
               'label': np.stack([b[1] for b in ds], axis=0)}
    
    def _prep_dl(self, ds):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last,
                          collate_fn=self._prep_ds)

    def get_datasets(self, train=True):
        cls_to_prep = np.random.choice(list(self.trn_cls_to_obj.keys() if train else self.test_cls_to_obj),
                                       self.n_way, replace=False)
        cls_map = {c: i for i, c in enumerate(cls_to_prep)}
        ds_trn = []
        ds_val = []
        raw_ds = self.trn_cls_to_obj if train else self.test_cls_to_obj
        for c in cls_to_prep:
            ids = np.random.choice(len(raw_ds[c]), self.n_shots + 10, replace=False)
            ds_trn.extend([(raw_ds[c][idx], cls_map[c]) for idx in ids[:self.n_shots]])
            ds_val.extend([(raw_ds[c][idx], cls_map[c]) for idx in ids[self.n_shots:]])
        return self._prep_dl(ds_trn), self._prep_ds(ds_val)


def prepare_datasets(ds_name='cifar100'):
    class ToNumpy:
        def __call__(self, pic):
            return np.asarray(pic.permute(1, 2, 0), dtype=np.float32)

    name_to_mean = {
                    'cifar100': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    'omniglot': (0.9157,)
                    }
    name_to_std = {
                   'cifar100': (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                    'omniglot': (0.2778,)
                   }
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=name_to_mean[ds_name], std=name_to_std[ds_name]),
        transforms.Resize(32 if ds_name == 'cifar100' else 28),
        ToNumpy()
        ])


    np.random.seed(0)

    if ds_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.Omniglot(root='./data', background=True, download=True, transform=transform)
    return trainset
