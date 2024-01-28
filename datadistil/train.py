import torch
import jax
import jax.numpy as jnp
import haiku as hk
from haiku.nets import ResNet18, MobileNetV1
from model import CNN
from train_state import create_train_state
from dataset import get_dataloaders
import numpy as np
from hpo_algs import *
from argparse import ArgumentParser
from tqdm.auto import tqdm
import json
from functools import partial


def parse_method(method: str):
    if method == 'DrMAD':
        return method, None
    elif 'proposed' in method:
        return 'proposed', float(method.split('_')[-1])
    elif 'IFT' in method:
        return 'IFT', (int(method.replace('IFT_', '')),)
    elif method == 'luketina':
        return 'luketina', None
    elif method == 'FISH':
        return 'FISH', None
    else:
        return ValueError('Unknorn method: ' + method)


@partial(jax.jit, static_argnums=3)
def outer_update(grad, opt_state, params, optimizer):
    updates, new_state = optimizer.update(grad, opt_state, params)
    new_p = optax.apply_updates(params, updates)
    return new_p, new_state


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, default=0)
    parser.add_argument('--T', type=int, required=False, default=20)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--data_size', type=int, required=False, default=1)
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
    conv_net = hk.transform_with_state(lambda x, t: eval(args.backbone)(num_classes=n_cls)(x, t))
    # conv_net = hk.transform(lambda x, t: eval(args.backbone)(num_classes=n_cls)(x, t))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, valloader, testloader = get_dataloaders(args.batch_size, args.dataset)

    # create a distilled dataset
    distil_imgs = np.random.randn(n_cls * args.data_size, *name_to_shape[args.dataset])
    distil_label = np.arange(n_cls).repeat(args.data_size)
    distil_batch = {'image': distil_imgs, 'label': distil_label}

    outer_opt = optax.adam(args.outer_lr)
    out_state = outer_opt.init(distil_imgs)

    method, m_params = parse_method(args.method)
    for outer_step in tqdm(range(args.outer_steps)):
        state = create_train_state(conv_net, jax.random.PRNGKey(seed), args.T * args.outer_steps, args.inner_lr,
                                   inp_shape=name_to_shape[args.dataset])
        x, y = next(iter(valloader))
        val_batch = {'image': x, 'label': y,
                     'lambda': jnp.zeros((y.shape[0],))}
        if method == 'proposed':
            state, g_so = proposed_so_grad(state, distil_batch, val_batch, m_params, args.T)
        elif method == 'DrMAD':
            state, g_so = drmad_grad(state, distil_batch, val_batch, args.T)
        elif method == 'IFT':
            state, g_so = IFT_grad(state, distil_batch, val_batch, *m_params, args.T)
        elif method == 'luketina':
            state, g_so = luketina_so_grad(state, distil_batch, val_batch, args.T)
        elif method == 'FISH':
            state, g_so = fish_so_grad(state, distil_batch, val_batch, args.T)
        else:
            raise ValueError(f'Unknown method: {method}')
        
        distil_imgs, out_state = outer_update(g_so, out_state, distil_imgs, outer_opt)

        # eval
        if outer_step % args.val_freq == 0 and outer_step > 0:
            for x, y in testloader:
                test_batch = {'image': x, 'label': y}
                state = compute_metrics(state=state, batch=test_batch)
        
            for metric,value in state.metrics.compute().items():
                metrics_history[seed][f'test_{metric}'].append(value.item())
            state = state.replace(metrics=state.metrics.empty())

    acc_arr = np.stack([metrics_history[s]['test_accuracy'] for s in [seed]], axis=0)
    print('Finished with', acc_arr.max(-1))
    with open(f'{args.method}_{seed}.json', 'w') as f:
        f.write(json.dumps({seed: float(acc_arr.max(-1).item())}))


if __name__ == '__main__':
    main()
