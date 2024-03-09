import numpy as np
import torchvision
from torchvision import transforms


class DataGenerator:
    def __init__(self, trn_set, test_set, n_way, n_shots, num_trn_cls=50,
                 num_val_cls=50, num_cls_total=100):
        self.n_way = n_way
        ids = np.random.choice(num_cls_total, num_trn_cls + num_val_cls, replace=False)
        self.trn_cls_to_obj = {d: [] for d in ids[:num_trn_cls]}
        self.test_cls_to_obj = {d: [] for d in ids[num_trn_cls:]}
        for x, y in trn_set:
            if y in self.trn_cls_to_obj:
                self.trn_cls_to_obj[y].append(x)
        for x, y in test_set:
            if y in self.test_cls_to_obj:
                self.test_cls_to_obj[y].append(x)
        self.n_shots = n_shots

    def _prep_ds(self, ds):
        return {'image': np.stack([b[0] for b in ds], axis=0),
               'label': np.stack([b[1] for b in ds], axis=0)}

    def get_datasets(self, train=True):
        cls_to_prep = np.random.choice(list(self.trn_cls_to_obj.keys() if train else self.test_cls_to_obj),
                                       self.n_way, replace=False)
        cls_map = {c: i for i, c in enumerate(cls_to_prep)}
        ds_trn = []
        ds_val = []
        raw_ds = self.trn_cls_to_obj if train else self.test_cls_to_obj
        for c in cls_to_prep:
            ids = np.random.choice(len(raw_ds[c]), 2 * self.n_shots, replace=False)
            ds_trn.extend([(raw_ds[c][idx], cls_map[c]) for idx in ids[:self.n_shots]])
            ds_val.extend([(raw_ds[c][idx], cls_map[c]) for idx in ids[self.n_shots:]])
        return self._prep_ds(ds_trn), self._prep_ds(ds_val)


def prepare_datasets():
    class ToNumpy:
        def __call__(self, pic):
            return np.asarray(pic.permute(1, 2, 0), dtype=np.float32)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ToNumpy()
        ])


    np.random.seed(0)

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return trainset, testset
