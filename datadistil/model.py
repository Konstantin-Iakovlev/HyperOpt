import jax
import haiku as hk


class CNN(hk.Module):
    def __init__(self, num_classes=10, channels=128):
        super().__init__(name="CNN")
        self.bn_aux = hk.BatchNorm(True, True, 0.9) # not used
        self.conv1 = hk.Conv2D(output_channels=channels, kernel_shape=(3,3), with_bias=False)
        self.bn1 = hk.InstanceNorm(True, True)
        self.conv2 = hk.Conv2D(output_channels=channels, kernel_shape=(3,3), with_bias=False)
        self.bn2 = hk.InstanceNorm(True, True)
        self.conv3 = hk.Conv2D(output_channels=channels, kernel_shape=(3,3), with_bias=False)
        self.bn3 = hk.InstanceNorm(True, True)
        self.flatten = hk.Flatten()
        self.linear1 = hk.Linear(channels)
        self.linear2 = hk.Linear(num_classes)

    def __call__(self, x_batch, is_training):
        x = self.conv1(x_batch)
        x = self.bn1(x)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        x = self.conv3(x)
        x = self.bn3(x)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x
