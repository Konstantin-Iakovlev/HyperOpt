import torch
import jax
import jax.numpy as jnp
import haiku as hk
from train_state import create_dw_train_state
from model import CNN
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
    parser.add_argument('--backbone', type=str, required=False, default='MobileNetV1')
    parser.add_argument('--inner_lr', type=float, required=False, default=1e-1)
    parser.add_argument('--outer_lr', type=float, required=False, default=1e-2)
    parser.add_argument('--val_freq', type=int, required=False, default=20)
    args = parser.parse_args()

    metrics_history = {seed: {'train_loss': [],
                    'train_accuracy': [],
                    'test_loss': [],
                    'test_accuracy': []} for seed in [args.seed]}

    conv_net = hk.transform_with_state(lambda x, t: eval('hk.nets.' + args.backbone)(num_classes=10)(x, t))
    wnet = hk.transform(lambda x: hk.nets.MLP([args.wnet_hidden, 1],
                                            activation=jax.nn.tanh, activate_final=False)(x))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = create_dw_train_state(conv_net, wnet, jax.random.PRNGKey(seed), args.T * args.outer_steps,
                                learning_rate=args.inner_lr, alpha_lr=args.outer_lr, input_shape=[32, 32, 3])

    trainloader, valloader, testloader = get_dataloaders_cifar(args.corruption, batch_size=args.batch_size)
    method, m_params = parse_method(args.method)
    for outer_step in tqdm(range(args.outer_steps)):
        
        x_val, y_val = next(iter(valloader))
        val_batch = {'image': jnp.asarray(x_val), 'label': jnp.asarray(y_val)}
        batches = []
        for i, (x, y) in enumerate(trainloader):
            if i >= args.T:
                break
            batches.append({'image': jnp.asarray(x), 'label': jnp.asarray(y)})
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
        else:
            raise ValueError('Unknown ' + method)
        
        state = state.apply_h_gradients(h_grads=g_so)

        # eval
        if outer_step % args.val_freq == 0 and outer_step > 0:
            for _, (x, y) in enumerate(testloader):
                val_batch = {'image': jnp.asarray(x), 'label': jnp.asarray(y)}
                state = compute_metrics(state=state, batch=val_batch)
            for metric,value in state.metrics.compute().items():
                metrics_history[seed][f'test_{metric}'].append(value.item())
    acc_arr = np.stack([metrics_history[s]['test_accuracy'] for s in [seed]], axis=0)
    print('Finished with', acc_arr.max(-1))
    with open(f'{args.method}_{seed}.json', 'w') as f:
        f.write(json.dumps({seed: float(acc_arr.max(-1).item())}))


if __name__ == '__main__':
    main()
