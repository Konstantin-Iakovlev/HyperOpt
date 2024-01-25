import jax
import jax.numpy as jnp
import haiku as hk


class UnetConvBlock(hk.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_1 = hk.Conv2D(out_channels, 3)
        self.conv_2 = hk.Conv2D(out_channels, 3)

    def __call__(self, x):
        out = self.conv_1(x)
        out = jax.nn.relu(out)
        out = self.conv_2(out)
        out = jax.nn.relu(out)
        return out


class UnetUpConvBlock(hk.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.upconv = hk.Conv2DTranspose(out_channels, 2, 2)
        self.conv_block = UnetConvBlock(out_channels)

    def __call__(self, x, bridge):
        up = self.upconv(x)
        out = jnp.concatenate([up, bridge], axis=-1)
        out = self.conv_block(out)
        return out


class Unet(hk.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()
        self.down_1 = UnetConvBlock(2 ** n_filters)
        self.down_2 = UnetConvBlock(2 ** (n_filters + 1))
        self.up_1 = UnetUpConvBlock(2 ** n_filters)
        self.last = hk.Conv2D(in_channels, kernel_shape=1)

    def __call__(self, x, rng):
        noise_channel = jax.random.normal(rng, shape=[*x.shape[:-1], 1])
        inp = jnp.concatenate([x, noise_channel], axis=-1)
        out_1 = self.down_1(inp)
        out_2 = hk.max_pool(out_1, 2, 2, 'VALID')
        out_2 = self.down_2(out_2)
        up_1 = self.up_1(out_2, out_1)
        return self.last(up_1) + x
