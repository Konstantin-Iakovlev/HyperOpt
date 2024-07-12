import haiku as hk
import jax


class CNN(hk.Module):
    def __init__(self, num_classes=10):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=6, kernel_shape=5, padding='SAME')
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=5, padding='VALID')
        self.flatten = hk.Flatten()
        self.lin1 = hk.Linear(84)
        self.logits = hk.Linear(num_classes, name='logits')

    def __call__(self, x_batch, is_training=True):
        x = self.conv1(x_batch)
        x = jax.nn.leaky_relu(x)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        
        x = self.conv2(x)
        x = jax.nn.leaky_relu(x)
        x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')

        x = self.flatten(x)
        x = jax.nn.leaky_relu(self.lin1(x))
        x = self.logits(x)
        return x
