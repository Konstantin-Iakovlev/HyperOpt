import torch
import jax
import jax.numpy as jnp
import haiku as hk
from train_state import create_nas_train_state
from model import CNN
from dataset import get_dataloaders
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
    elif method == 'fo':
        return 'fo', None
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
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--channels', type=int, required=False, default=16)
    parser.add_argument('--outer_steps', type=int, required=False, default=20_000)
    parser.add_argument('--method', type=str, required=True, default='luketina')
    parser.add_argument('--dataset', type=str, required=False, default='cifar10')
    parser.add_argument('--inner_lr', type=float, required=False, default=0.025)
    parser.add_argument('--outer_lr', type=float, required=False, default=3e-4)
    parser.add_argument('--val_freq', type=int, required=False, default=20)
    args = parser.parse_args()

    metrics_history = {seed: {'train_loss': [],
                    'train_accuracy': [],
                    'test_loss': [],
                    'test_accuracy': []} for seed in [args.seed]}

    name_to_cls = {'cifar10': 10,
                   'cifar100': 100,
                   'svhn': 10,
                   'fmnist': 10,
                   }
    name_to_shape = {'cifar10': [32, 32, 3],
                    'cifar100': [32, 32, 3],
                    'svhn': [32, 32, 3],
                    'fmnist': [28, 28, 1]}
    n_cls = name_to_cls[args.dataset]
    cnn = hk.transform_with_state(lambda x, t: CNN(args.channels, n_cls, 1) (x, t))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = create_nas_train_state(cnn, jax.random.PRNGKey(seed), args.T * args.outer_steps,
                                   learning_rate=args.inner_lr, alpha_lr=args.outer_lr,
                                   inp_shape=name_to_shape[args.dataset])

    trainloader, valloader, testloader = get_dataloaders(args.batch_size, args.dataset)
    method, m_params = parse_method(args.method)
    for outer_step in tqdm(range(args.outer_steps)):
        x_val, y_val = next(iter(valloader))
        val_batch = {'image': x_val, 'label': y_val.astype(np.int32)}
        batches = []
        for i, (x, y) in enumerate(trainloader):
            if i >= args.T:
                break
            batches.append({'image': x, 'label': y.astype(np.int32)})
        assert len(batches) == args.T, f'{len(batches)} < {args.T}. Consider bigger data size'
        if method == 'proposed':
            state, g_so = proposed_so_grad(state, batches, val_batch, m_params)
        elif method == 'DrMAD':
            state, g_so = drmad_grad(state, batches, val_batch)
        # elif method == 'IFT':
        #     state, g_so = IFT_grad(state, batches, val_batch, m_params)
        elif method == 'fo':
            for batch in batches:
                state = inner_step(state, batch)
            g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
        elif method == 'luketina':
            state, g_so = luketina_so_grad(state, batches, val_batch)
        # elif method == 'FISH':
        #     state, g_so = fish_so_grad(state, batches, val_batch)
        else:
            raise ValueError('Unknown ' + method)
        
        g_fo = fo_grad(state, val_batch)
        state = state.apply_h_gradients(h_grads=jax.tree_util.tree_map(lambda x, y: x + y, g_fo, g_so))

        # eval
        if outer_step % args.val_freq == 0 and outer_step > 0:
            for _, (x, y) in enumerate(testloader):
                val_batch = {'image': x, 'label': y.astype(np.int32)}
                state = compute_metrics(state=state, batch=val_batch)
            for metric,value in state.metrics.compute().items():
                metrics_history[seed][f'test_{metric}'].append(value.item())
            state = state.replace(metrics=state.metrics.empty())

    acc_arr = np.stack([metrics_history[s]['test_accuracy'] for s in [seed]], axis=0)
    print('Finished with', acc_arr.max(-1))
    with open(f'{args.method}_{seed}.json', 'w') as f:
        f.write(json.dumps({seed: float(acc_arr.max(-1).item())}))


if __name__ == '__main__':
    main()
