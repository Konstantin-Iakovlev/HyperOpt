import torch
import jax
import jax.numpy as jnp
import haiku as hk
from model import CNN
from haiku.nets import *
from train_state import create_bilevel_train_state
from dataset import prepare_datasets, DataGenerator
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
    elif method == 'fo':
        return 'fo', None
    else:
        raise ValueError('Unknorn method: ' + method)


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, default=0)
    parser.add_argument('--backbone', type=str, required=False, default='CNN')
    parser.add_argument('--train_classes', type=int, required=False, default=50)
    parser.add_argument('--val_classes', type=int, required=False, default=50)
    parser.add_argument('--num_ways', type=int, required=True, default=5)
    parser.add_argument('--num_shots', type=int, required=True, default=1)
    parser.add_argument('--T', type=int, required=False, default=10)
    parser.add_argument('--meta_batch_size', type=int, required=False, default=4)
    parser.add_argument('--outer_steps', type=int, required=False, default=1000)
    parser.add_argument('--method', type=str, required=True, default='proposed_0.999')
    parser.add_argument('--inner_lr', type=float, required=False, default=1e-1)
    parser.add_argument('--outer_lr', type=float, required=False, default=1e-3)
    parser.add_argument('--val_freq', type=int, required=False, default=20)
    args = parser.parse_args()

    metrics_history = {seed: {'train_loss': [],
                              'train_accuracy': [],
                              'test_loss': [],
                              'test_accuracy': []} for seed in [args.seed]}
    conv_net = hk.transform_with_state(lambda x, t: eval(args.backbone)(num_classes=args.num_ways)(x, t))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainset, testset = prepare_datasets()
    data_gen = DataGenerator(trainset, testset, args.num_ways, args.num_shots, args.train_classes, args.val_classes)
    method, m_params = parse_method(args.method)

    state = create_bilevel_train_state(conv_net, jax.random.PRNGKey(seed),
                                       args.inner_lr, args.outer_lr, out_steps=args.outer_steps)
    
    meta_grad = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    for outer_step in tqdm(range(args.meta_batch_size * args.outer_steps)):
        params, _ = conv_net.init(jax.random.PRNGKey(seed), jnp.ones([1, 32, 32, 3]), True)
        w_params, _ = hk.data_structures.partition(lambda m, n, p: 'warp' not in m, params)
        inn_state = state.inner_opt.init(w_params)
        state = state.replace(w_params=w_params, inner_opt_state=inn_state)
        
        ds_trn, ds_val = data_gen.get_datasets()
    
        val_batch = ds_val
        batches = [ds_trn] * args.T
        if method == 'proposed':
            state, g_so = proposed_so_grad(state, batches, val_batch, m_params)
        elif method == 'fo':
            for batch in batches:
                state = inner_step(state, batch)
            g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
        elif method == 'IFT':
            state, g_so = IFT_grad(state, batches, val_batch, m_params)
        elif method == 'luketina':
            state, g_so = luketina_so_grad(state, batches, val_batch)
        elif method == 'DrMAD':
            state, g_so = drmad_grad(state, batches, val_batch)
        else:
            raise ValueError('Unknown ' + method)

        g_fo = fo_grad(state, val_batch)
        total_g = jax.tree_util.tree_map(lambda x, y: x + y, g_fo, g_so)
        meta_grad = jax.tree_util.tree_map(lambda x, y: x + y / args.meta_batch_size, meta_grad, total_g)
        if (outer_step + 1) % args.meta_batch_size == 0:
            state = state.apply_h_gradients(h_grads=meta_grad)
            meta_grad = jax.tree_util.tree_map(jnp.zeros_like, meta_grad)

        # eval
        if outer_step % (args.val_freq * args.meta_batch_size) == 0 and outer_step > 0:
            for _ in range(50):
                params, _ = conv_net.init(jax.random.PRNGKey(seed), jnp.ones([1, 32, 32, 3]), True)
                w_params, _ = hk.data_structures.partition(lambda m, n, p: 'warp' not in m, params)
                inn_state = state.inner_opt.init(w_params)
                state = state.replace(w_params=w_params, inner_opt_state=inn_state)

                ds_trn, ds_val = data_gen.get_datasets(False)
                for _ in range(args.T):
                    state = inner_step(state, ds_trn)
                state = compute_metrics(state=state, batch=ds_val)
                
            for metric,value in state.metrics.compute().items():
                metrics_history[seed][f'test_{metric}'].append(value.item())
            state = state.replace(metrics=state.metrics.empty())

    acc_arr = np.stack([metrics_history[s]['test_accuracy'] for s in [seed]], axis=0)
    print('Finished with', acc_arr.max(-1))
    print(acc_arr)
    with open(f'{args.method}_{seed}.json', 'w') as f:
        f.write(json.dumps({seed: float(acc_arr.max(-1).item())}))


if __name__ == '__main__':
    main()
