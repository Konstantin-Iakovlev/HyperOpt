import torch
import jax
import jax.numpy as jnp
import haiku as hk
from train_state import create_train_state
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
        return 'IFT', *list(map(int, method.split('_')[-2:]))
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
    parser.add_argument('--corruption', type=float, required=True, default=0.0)
    parser.add_argument('--T', type=int, required=False, default=20)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--outer_steps', type=int, required=False, default=1000)
    parser.add_argument('--wnet_hidden', type=int, required=False, default=100)
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
    n_cls = int(args.dataset.replace('cifar', ''))
    conv_net = hk.transform_with_state(lambda x, t: eval('hk.nets.' + args.backbone)(num_classes=n_cls)(x, t))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = create_train_state(conv_net, jax.random.PRNGKey(seed), args.T * args.outer_steps, args.inner_lr)
    trainloader, valloader, testloader = get_dataloaders_cifar(args.corruption, args.batch_size, args.dataset)
    w_logits = jnp.zeros([(len(trainloader) + 1) * args.batch_size,], dtype=jnp.float32)
    outer_opt = optax.adam(1.0)
    out_state = outer_opt.init(w_logits)
    method, m_params = parse_method(args.method)
    for outer_step in tqdm(range(args.outer_steps)):
        x, y = next(iter(valloader))
        val_batch = {'image': jnp.asarray(x), 'label': jnp.asarray(y),
                     'lambda': jnp.zeros((y.shape[0],))}
        batches = []
        for i, (x, y, ids) in enumerate(trainloader):
            ids = jnp.asarray(ids)
            x = jnp.asarray(x)
            y = jnp.asarray(y)
            batches.append({'image': x, 'label': y,
                            'lambda': w_logits[ids], 'ids': ids})
            if i == args.T - 1:
                break
        if method == 'proposed':
            state, g_so_arr = proposed_so_grad(state, batches, val_batch, m_params)
        elif method == 'DrMAD':
            state, g_so_arr = drmad_grad(state, batches, val_batch)
        elif method == 'IFT':
            state, g_so_arr = IFT_grad(state, batches, val_batch, *m_params)
        elif method == 'baseline':
            g_so_arr = [jnp.zeros([args.batch_size,], dtype=jnp.float32)] * args.T
            for batch in batches:
                state = inner_step_baseline(state, batch)
        elif method == 'luketina':
            state, g_so_arr = luketina_so_grad(state, batches, val_batch)
        elif method == 'FISH':
            state, g_so_arr = fish_so_grad(state, batches, val_batch)
        else:
            raise ValueError(f'Unknown method: {method}')
        
        outer_grad = jnp.zeros_like(w_logits)
        for batch, g_so in zip(batches, g_so_arr):
            ids = batch['ids']
            outer_grad = outer_grad.at[ids].set(outer_grad[ids] + g_so)
        updates, out_state = outer_opt.update(outer_grad, out_state, w_logits)
        w_logits = optax.apply_updates(w_logits, updates)

        # eval
        if outer_step % args.val_freq == 0 and outer_step > 0:
            for x, y in testloader:
                test_batch = {'image': jnp.asarray(x), 'label': jnp.asarray(y)}
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
