import jax
import haiku as hk


class CNN(hk.Module):
    def __init__(self, num_classes=10):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=6, kernel_shape=5, with_bias=True, padding='SAME')
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=5, with_bias=True, padding='VALID')
        self.flatten = hk.Flatten()
        self.linear1 = hk.Linear(120)
        self.linear2 = hk.Linear(84)
        self.linear3 = hk.Linear(num_classes)

    def __call__(self, x_batch, is_training):
        x = self.conv1(x_batch)
        x = jax.nn.leaky_relu(x)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        
        x = self.conv2(x)
        x = jax.nn.leaky_relu(x)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')

        x = self.flatten(x)
        x = jax.nn.leaky_relu(self.linear1(x))
        x = jax.nn.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x
