import jax
import jax.numpy as jnp
import haiku as hk
from ops import *


class LayerChoice(hk.Module):
    def __init__(self, channels, stride, label='none'):
        super().__init__()
        self.label = label
        self.ops = [
            PoolBN('max', 3, stride, affine=False),
            # PoolBN('avg', 3, stride, affine=False),
            # Identity() if stride == 1 else FactorizedReduce(channels, False),
            SepConv(channels, channels, 3, stride, False),
            # SepConv(channels, channels, 5, stride, False),
            # DilConv(channels, channels, 3, stride, 2, False),
            # DilConv(channels, channels, 5, stride, 2, False)
        ]
        # self.op_names = ('maxpool', 'avgpool', 'skipconnect', 'sepconv3x3',
        #                  'sepconv5x5', 'dilconv3x3', 'dilconv5x5')
        self.op_names = ('maxpool', 'sepconv3x3')
        
    def __call__(self, x, is_training):
        """x: (bs, w, h, c)"""
        alpha = hk.get_parameter("lc_alpha", shape=(len(self.ops),), init=hk.initializers.RandomNormal(1e-3))
        res = jnp.stack([op(x, is_training) for op in self.ops], axis=0)  # (# prev, bs, w, h, c)
        weights = jax.nn.softmax(alpha, axis=-1).reshape(-1, 1, 1, 1, 1)
        return (res * weights).sum(0)
        

class InputChoice(hk.Module):
    def __init__(self, n_cand: int, n_chosen: int, label='none'):
        super().__init__()
        self.n_chosen = n_chosen
        self.label = label
        self.n_cand = n_cand

    def __call__(self, inputs):
        alpha = hk.get_parameter("ic_alpha", shape=(self.n_cand,), dtype=jnp.float32,
                                 init=hk.initializers.RandomNormal(1e-3))
        inputs = jnp.stack(inputs, axis=0)  # (#cand, bs, w, h, c)
        weights = jax.nn.softmax(alpha, axis=-1).reshape(-1, 1, 1, 1, 1)
        return (inputs * weights).sum(0)


class Node(hk.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect, drop_path_prob):
        super().__init__()
        choice_keys = []
        self.edges = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append(f'{node_id}_p{i}')
            self.edges.append(LayerChoice(channels, stride, choice_keys[-1]))
        self.drop_path = DropPath(drop_path_prob)
        self.input_switch = InputChoice(n_cand=len(choice_keys), n_chosen=2, label=f'{node_id}_switch')

    def __call__(self, prev_nodes, is_training):
        """Prev nodes: List [bs, w, h, c]"""
        out = [self.drop_path(edge(node, is_training), is_training) \
               for edge, node in zip(self.edges, prev_nodes)]
        return self.input_switch(out)


### TEST
def test_layer_choice():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 32, 32, 16))
    lc = hk.transform_with_state(lambda x, is_t: LayerChoice(16, 1)(x, is_t))
    params, state = lc.init(rng, x, True)
    # print(params.keys())
    out = lc.apply(params, state, rng, x, True)[0]  # out, state_new = apply 
    assert out.shape == x.shape


def test_inp_choice():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 32, 32, 16))
    ic = hk.transform_with_state(lambda y: InputChoice(3, 2)(y))
    params, state = ic.init(rng, [x, x, x])
    out = ic.apply(params, state, rng, [x, x, x])[0]  # out, state_new = apply 
    assert out.shape == x.shape


def test_node():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 32, 32, 16))
    node = hk.transform_with_state(lambda x, t: Node(3, 3, 16, 0, 0.1)(x, t))
    params, state = node.init(rng, [x, x, x], True)
    out = node.apply(params, state, rng, [x, x, x], True)[0]
    assert out.shape == x.shape, f'{out.shape}'


# test_layer_choice()
# test_inp_choice()
# test_node()


class Cell(hk.Module):
    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction, drop_path_prob):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        if reduction_p:
            self.preproc0 = FactorizedReduce(channels, affine=False)
        else:
            self.preproc0 = StdConv(channels, 1, 1, affine=False)
        self.preproc1 = StdConv(channels, 1, 1, affine=False)

        # generate dag
        self.mutable_ops = []
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node(f"{'reduce' if reduction else 'normal'}_n{depth}",
                                         depth, channels, 2 if reduction else 0, drop_path_prob))

    def __call__(self, s0, s1, is_training):
        inputs = [self.preproc0(s0, is_training), self.preproc1(s1, is_training)]
        for node in self.mutable_ops:
            out = node(inputs, is_training)
            inputs.append(out)
        out = jnp.concatenate(inputs[2:], axis=-1)  # (bs, w, h, c * nodes)
        return out


def test_cell():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 32, 32, 16))
    cell = hk.transform_with_state(lambda x, y, t: Cell(4, 3*16, 3 * 16, 16, False, False, 0.1)(x, y, t))
    params, state = cell.init(rng, x, x, True)
    out = cell.apply(params, state, rng, x, x, True)[0]
    assert out.shape == (3, 32, 32, 64), f'{out.shape}'


# test_cell()


class CNN(hk.Module):
    def __init__(self, channels, n_classes, n_layers, n_nodes=4,
                stem_multiplier=3, drop_path_prob=0.0):
        # TODO: add aux if necessary
        # TODO: share alphas among cells
        super().__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        c_cur = stem_multiplier * self.channels
        self.stem = hk.Conv2D(c_cur, 3, stride=1, padding='SAME', with_bias=False)
        self.stem_bn = hk.BatchNorm(True, True, 0.9)

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = []
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True
            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction, drop_path_prob)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out
            
        self.linear = hk.Linear(n_classes)


    def __call__(self, x, is_training):
        s0 = s1 = self.stem_bn(self.stem(x), is_training)
        for i, cell in enumerate(self.cells):
            # cell(s0, s1)
            s0, s1 = s1, cell(s0, s1, is_training)

        out = s1.mean(axis=[-1, -2])  # global adaptive pooling
        out = out.reshape(out.shape[0], -1)  # flatten
        logits = self.linear(out)
        return logits

def test_cnn():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 32, 32, 3))
    cnn = hk.transform_with_state(lambda x, t: CNN(16, 10, 1)(x, t))
    params, state = cnn.init(rng, x, True)
    out = cnn.apply(params, state, rng, x, True)[0]
    assert out.shape == (3, 10)


# test_cnn()
