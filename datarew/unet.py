from typing import Optional
import jax
import haiku as hk


class Unet(hk.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv_1 = hk.Conv2D(channels, 3, 1)
        self.pool_1 = hk.MaxPool(3, 1, padding='SAME')
        self.conv_2 = hk.Conv2D(channels * 2, 3, 1)
        self.conv_3 = hk.Conv2DTranspose(channels, 3, 1)
        self.conv_4 = hk.Conv2DTranspose(in_channels, 3, 1)
    
    def __call__(self, x):
        x_1 = self.pool_1(jax.nn.relu(self.conv_1(x)))
        x_2 = jax.nn.relu(self.conv_2(x_1))
        x_3 = jax.nn.relu(self.conv_3(x_2)) + x_1
        x_4 = jax.nn.relu(self.conv_4(x_3)) + x
        return x_4

