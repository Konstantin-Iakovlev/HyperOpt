import torch
import jax
import jax.numpy as jnp
import haiku as hk
from train_state import create_train_state
from model import CNN
from haiku.nets import *
from dataset import get_dataloaders_cifar
import numpy as np
from hpo_algs import *
from argparse import ArgumentParser
from tqdm.auto import tqdm
import json


def parse_method(method: str):
    if method == 'DrMAD':
        return method, None
    elif 'proposed' in method:
        return 'proposed', float(method.split('_')[-1])
    elif 'IFT' in method:
        return 'IFT', int(method.split('_')[-1])
    elif method == 'baseline':
        return 'baseline', None
    elif method == 'luketina':
        return 'luketina', None
    elif method == 'FISH':
        return 'FISH', None
    else:
        return ValueError('Unknorn method: ' + method)


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, default=0)
    parser.add_argument('--T', type=int, required=False, default=20)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--outer_steps', type=int, required=False, default=1000)
    parser.add_argument('--method', type=str, required=True, default='proposed_0.999')
    parser.add_argument('--backbone', type=str, required=False, default='ResNet18')
    parser.add_argument('--dataset', type=str, required=False, default='cifar10')
    parser.add_argument('--inner_lr', type=float, required=False, default=1e-1)
    parser.add_argument('--outer_lr', type=float, required=False, default=1e-2)
    parser.add_argument('--val_freq', type=int, required=False, default=20)
    args = parser.parse_args()

    metrics_history = {seed: {'train_loss': [],
                    'train_accuracy': [],
                    'test_loss': [],
                    'test_accuracy': [],
                    'learning_rate': []} for seed in [args.seed]}

    n_cls = 100 if args.dataset == 'cifar100' else 10
    name_to_shape = {'cifar10': [32, 32, 3],
                    'cifar100': [32, 32, 3],
                    'svhn': [32, 32, 3],
                    'fmnist': [28, 28, 1],
                    'mnist': [28, 28, 1]
                    }
    conv_net = hk.transform_with_state(lambda x, t: eval(args.backbone)(num_classes=n_cls)(x, t))
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = create_train_state(conv_net, jax.random.PRNGKey(seed),
                                learning_rate=args.inner_lr, alpha_lr=args.outer_lr,
                                input_shape=name_to_shape[args.dataset])

    trainloader, valloader, testloader = get_dataloaders_cifar(batch_size=args.batch_size,
                                                               ds_name=args.dataset)
    method, m_params = parse_method(args.method)
    for outer_step in tqdm(range(args.outer_steps)):
        
        x_val, y_val = next(iter(valloader))
        val_batch = {'image': x_val, 'label': y_val}
        batches = []
        for i, (x, y) in enumerate(trainloader):
            if i >= args.T:
                break
            batches.append({'image': x, 'label': y})
        assert len(batches) == args.T, f'{len(batches)} < {args.T}. Consider bigger data size'
        if method == 'proposed':
            state, g_so = proposed_so_grad(state, batches, val_batch, m_params)
        elif method == 'baseline':
            g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
            for batch in batches:
                state = inner_step(state, batch)
        elif method == 'luketina':
            state, g_so = luketina_so_grad(state, batches, val_batch)
        else:
            raise ValueError('Unknown ' + method)
        
        if method != 'baseline':
            state = state.apply_h_gradients(h_grads=g_so)

        # eval
        if outer_step % args.val_freq == 0 and outer_step > 0:
            for _, (x, y) in enumerate(testloader):
                val_batch = {'image': x, 'label': y}
                state = compute_metrics(state=state, batch=val_batch)
            for metric,value in state.metrics.compute().items():
                metrics_history[seed][f'test_{metric}'].append(value.item())
            state = state.replace(metrics=state.metrics.empty())
            metrics_history[seed]['learning_rate'].append(state.h_params)
    acc_arr = np.stack([metrics_history[s]['test_accuracy'] for s in [seed]], axis=0)
    print('Finished with', acc_arr.max(-1))
    print(acc_arr)
    print([lr.item() for lr in metrics_history[seed]['learning_rate']])
    with open(f'{args.method}_{seed}.json', 'w') as f:
        f.write(json.dumps({seed: float(acc_arr.max(-1).item())}))


if __name__ == '__main__':
    main()