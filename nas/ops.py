import jax
import jax.numpy as jnp
import haiku as hk

class DropPath(hk.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def __call__(self, x, is_training=False):
        if is_training and self.p > 0:
            keep_p = 1.0 - self.p
            mask = jax.random.bernoulli(hk.next_rng_key(), keep_p, (x.shape[0], 1, 1, 1))
            return x / keep_p * mask
        return x


class Identity(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, is_training=False):
        return x


class PoolBN(hk.Module):
    def __init__(self, pool_type, kernel_size, stride, affine=True):  # C is not needed
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = hk.MaxPool(window_shape=kernel_size, strides=stride, padding='SAME')
        elif pool_type.lower() == 'avg':
            self.pool = hk.AvgPool(window_shape=kernel_size, strides=stride, padding='SAME')
            # TODO: count_include_pad=False
        else:
            raise ValueError()
        self.bn = hk.BatchNorm(affine, affine, 0.9)

    def __call__(self, x, is_training):
        out = self.pool(x)
        out = self.bn(out, is_training)
        return out


class StdConv(hk.Module):
    def __init__(self, C_out, kernel_size, stride, affine=True):  # C_in is not needed
        super().__init__()
        self.conv = hk.Conv2D(C_out, kernel_shape=kernel_size, stride=stride,
                              padding='SAME', with_bias=False)
        self.bn = hk.BatchNorm(affine, affine, 0.9)

    def __call__(self, x, is_training):
        out = jax.nn.relu(x)
        out = self.conv(out)
        out = self.bn(out, is_training)
        return out


class FacConv(hk.Module):
    """
    Factorized conv: ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, affine=True):
        super().__init__()
        self.conv1 = hk.Conv2D(C_in, kernel_shape=(kernel_length, 1), stride=stride, padding='SAME',
                              with_bias=False)
        self.conv2 = hk.Conv2D(C_out, kernel_shape=(1, kernel_length), stride=stride, padding='SAME',
                               with_bias=False)
        self.bn = hk.BatchNorm(affine, affine, 0.9)
        
    def __call__(self, x, is_training):
        out = jax.nn.relu(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.bn(out, is_training)
        return out


class DilConv(hk.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, dilation, affine=True):
        super().__init__()
        self.conv1 = hk.Conv2D(C_in, kernel_shape=kernel_size, stride=stride, rate=dilation,
                               padding='SAME', with_bias=False)
        self.conv2 = hk.Conv2D(C_out, kernel_shape=1, stride=1, rate=dilation,
                               padding='SAME', with_bias=False)
        self.bn = hk.BatchNorm(affine, affine, 0.9)
        
    def __call__(self, x, is_training):
        out = jax.nn.relu(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.bn(out, is_training)
        return out


class SepConv(hk.Module):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, affine=True):
        super().__init__()
        self.conv1 = DilConv(C_in, C_out, kernel_size, stride, 1, affine)
        self.conv2 = DilConv(C_in, C_out, kernel_size, 1, 1, affine)

    def __call__(self, x, is_training):
        return self.conv2(self.conv1(x, is_training), is_training)
        

class FactorizedReduce(hk.Module):
    """
    Reduce feature map size by factorized pointwise (stride=2).
    """
    def __init__(self, C_out, affine=True):
        super().__init__()
        self.conv1 = hk.Conv2D(C_out // 2, kernel_shape=1, stride=2, padding='VALID', with_bias=False)
        self.conv2 = hk.Conv2D(C_out // 2, kernel_shape=1, stride=2, padding='VALID', with_bias=False)
        self.bn = hk.BatchNorm(affine, affine, 0.9)

    def __call__(self, x, is_training):
        out = jax.nn.relu(x)
        out = jnp.concatenate([self.conv1(out), self.conv2(out[:, 1:, 1:])], axis=-1)
        out = self.bn(out, is_training)
        return out
        
        
        
### TEST
def test_dp():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 2, 1, 1))
    dp = hk.transform(lambda x, is_training: DropPath(0.5)(x, is_training))
    params = dp.init(rng, x, True)
    assert dp.apply(params, rng, x, True).shape == x.shape


def test_id():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 2, 1, 1))
    id_ = hk.transform(lambda x, is_training: Identity()(x, is_training))
    params = id_.init(rng, x, True)
    assert id_.apply(params, None, x, True).shape == x.shape


def test_pool():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 16, 16, 3))
    pool = hk.transform_with_state(lambda x, is_t: PoolBN('avg', 3, 1)(x, is_t))
    params, state = pool.init(rng, x, True)
    assert pool.apply(params, state, rng, x, True)[0].shape == x.shape  # out, state_new = apply    


def test_std_conv():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 16, 16, 3))
    conv = hk.transform_with_state(lambda x, is_t: StdConv(3, 3, 1)(x, is_t))
    params, state = conv.init(rng, x, True)
    assert conv.apply(params, state, rng, x, True)[0].shape == x.shape  # out, state_new = apply

def test_fac_conv():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 16, 16, 3))
    conv = hk.transform_with_state(lambda x, is_t: FacConv(3, 3, 3, 1)(x, is_t))
    params, state = conv.init(rng, x, True)
    assert conv.apply(params, state, rng, x, True)[0].shape == x.shape  # out, state_new = apply 


def test_dil_conv():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 32, 32, 16))
    conv = hk.transform_with_state(lambda x, is_t: DilConv(16, 16, 5, 1, 2, False)(x, is_t))
    params, state = conv.init(rng, x, True)  # out, state_new = apply
    out = conv.apply(params, state, rng, x, True)[0]
    assert out.shape == x.shape, f'{out.shape}, {x.shape}'


def test_sep_conv():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 32, 32, 16))
    conv = hk.transform_with_state(lambda x, is_t: SepConv(16, 16, 3, 1)(x, is_t))
    params, state = conv.init(rng, x, True)
    assert conv.apply(params, state, rng, x, True)[0].shape == x.shape  # out, state_new = apply 


def test_fact_reduce():
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((3, 16, 16, 3))
    conv = hk.transform_with_state(lambda x, is_t: FactorizedReduce(4)(x, is_t))
    params, state = conv.init(rng, x, True)
    out = conv.apply(params, state, rng, x, True)[0]  # out, state_new = apply 
    assert out.shape == (3, 8, 8, 4), f'{out.shape}'


# test_dp()
# test_id()
# test_pool()
# test_std_conv()
# test_fac_conv()
# test_dil_conv()
# test_sep_conv()
# test_fact_reduce()
