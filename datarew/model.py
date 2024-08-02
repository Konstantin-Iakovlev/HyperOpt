import jax
import haiku as hk


class CNN(hk.Module):
    def __init__(self, num_classes=10):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=8, kernel_shape=3, with_bias=False)
        self.bn1 = hk.BatchNorm(True, True, 0.9)
        self.conv2 = hk.Conv2D(output_channels=8, kernel_shape=3, with_bias=False)
        self.bn2 = hk.BatchNorm(True, True, 0.9)
        self.conv3 = hk.Conv2D(output_channels=8, kernel_shape=3, with_bias=False)
        self.bn3 = hk.BatchNorm(True, True, 0.9)
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(num_classes)

    def __call__(self, x_batch, is_training):
        x = self.conv1(x_batch)
        x = self.bn1(x, is_training)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = jax.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, is_training)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = jax.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, is_training)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        x = jax.nn.relu(x)
        
        x = self.flatten(x)
        x = jax.nn.relu(x)
        x = self.linear(x)
        return x
