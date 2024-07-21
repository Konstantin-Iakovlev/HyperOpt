from typing import Any
import haiku as hk
import jax


class WarpLayer(hk.Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=3, with_bias=False)
        # self.conv2 = hk.Conv2D(output_channels=32, kernel_shape=3, with_bias=False)
    
    def __call__(self, x_batch, is_training=True):
        x = self.conv1(x_batch)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(x)

        # x = self.conv2(x)
        # x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        # x = jax.nn.relu(x)

        return x


class CNN(hk.Module):
    def __init__(self, num_classes=10):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=3, with_bias=False)
        self.warp1 = WarpLayer(name='warp1')
        self.conv2 = hk.Conv2D(output_channels=32, kernel_shape=3, with_bias=False)
        self.warp2 = WarpLayer(name='warp2')
        self.conv3 = hk.Conv2D(output_channels=32, kernel_shape=3, with_bias=False)
        self.warp3 = WarpLayer(name='warp3')
        self.flatten = hk.Flatten()
        self.logits = hk.Linear(num_classes, name='logits')

    def __call__(self, x_batch, is_training=True):
        x = self.conv1(x_batch)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = jax.nn.relu(x)

        x = self.warp1(x)
        
        x = self.conv2(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = jax.nn.relu(x)

        x = self.warp2(x)

        x = self.conv3(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = jax.nn.relu(x)

        x = self.warp3(x)
        
        x = self.flatten(x)
        x = self.logits(x)
        return x
