import torch
import jax
import jax.numpy as jnp
import haiku as hk
from haiku.nets import *
from train_state import create_dw_train_state
from model import CNN, ResNet9
from wide_res_net import WideResNet
from unet import Unet
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
    parser.add_argument('--T', type=int, required=False, default=10)
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--data_size', type=int, required=False, default=1_000_000)
    parser.add_argument('--outer_steps', type=int, required=False, default=2001)
    parser.add_argument('--method', type=str, required=True, default='proposed_0.999')
    parser.add_argument('--backbone', type=str, required=False, default='ResNet18')
    parser.add_argument('--dataset', type=str, required=False, default='svhn')
    parser.add_argument('--inner_lr', type=float, required=False, default=1e-2)
    parser.add_argument('--outer_lr', type=float, required=False, default=1e-4)
    parser.add_argument('--val_freq', type=int, required=False, default=500)
    args = parser.parse_args()

    metrics_history = {seed: {'train_loss': [],
                    'train_accuracy': [],
                    'test_loss': [],
                    'test_accuracy': []} for seed in [args.seed]}

    name_to_cls = {'cifar10': 10,
                   'cifar100': 100,
                   'svhn': 10,
                   'fmnist': 10,
                   'mnist': 10,
                   }
    name_to_shape = {'cifar10': [32, 32, 3],
                    'cifar100': [32, 32, 3],
                    'svhn': [32, 32, 3],
                    'fmnist': [28, 28, 1],
                    'mnist': [28, 28, 1],
                    }
    n_cls = name_to_cls[args.dataset]
    if args.backbone == 'WideResNet':
        conv_net = hk.transform_with_state(lambda x, t: WideResNet(num_classes=n_cls,
                                                                   depth=16, width=2)(x, is_training=t))
    else:
        conv_net = hk.transform_with_state(lambda x, t: eval(args.backbone)(num_classes=n_cls)(x, t))
    unet = hk.transform(lambda x, r: Unet(name_to_shape[args.dataset][-1], 3)(x, r))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = create_dw_train_state(conv_net, unet, jax.random.PRNGKey(seed), args.T * args.outer_steps,
                                learning_rate=args.inner_lr, alpha_lr=args.outer_lr, input_shape=name_to_shape[args.dataset])

    trainloader, valloader, testloader = get_dataloaders(batch_size=args.batch_size,
                                                        num_samples=args.data_size,
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
        elif method == 'DrMAD':
            state, g_so = drmad_grad(state, batches, val_batch)
        elif method == 'IFT':
            state, g_so = IFT_grad(state, batches, val_batch, m_params)
        elif method == 'baseline':
            g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
            for batch in batches:
                state = inner_step_baseline(state, batch)
        elif method == 'luketina':
            state, g_so = luketina_so_grad(state, batches, val_batch)
        elif method == 'FISH':
            state, g_so = fish_so_grad(state, batches, val_batch)
        else:
            raise ValueError('Unknown ' + method)
        
        state = state.apply_h_gradients(h_grads=g_so)

        # eval
        if outer_step % args.val_freq == 0 and outer_step > 0:
            for _, (x, y) in enumerate(testloader):
                val_batch = {'image': x, 'label': y}
                state = compute_metrics(state=state, batch=val_batch)
            for metric,value in state.metrics.compute().items():
                metrics_history[seed][f'test_{metric}'].append(value.item())
            state = state.replace(metrics=state.metrics.empty())
    acc_arr = np.stack([metrics_history[s]['test_accuracy'] for s in [seed]], axis=0)
    print(acc_arr)
    print('Finished with', acc_arr.max(-1))
    with open(f'{args.method}_{seed}.json', 'w') as f:
        f.write(json.dumps({seed: float(acc_arr.max(-1).item())}))


if __name__ == '__main__':
    main()
