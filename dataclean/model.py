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
        self.linear1 = hk.Linear(32)
        self.linear2 = hk.Linear(num_classes)

    def __call__(self, x_batch, is_training):
        x = self.conv1(x_batch)
        x = self.bn1(x, is_training)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=2, strides=2, padding='VALID')
        
        x = self.conv2(x)
        x = self.bn2(x, is_training)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=2, strides=2, padding='VALID')

        x = self.conv3(x)
        x = self.bn3(x, is_training)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=2, strides=2, padding='VALID')
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


def conv_block(out_channels: int, pool: bool = False):
    layers = [hk.Conv2D(out_channels, kernel_shape=3, padding='SAME'),
              hk.BatchNorm(True, True, 0.9), jax.nn.relu]
    if pool:
        layers.append(hk.MaxPool(window_shape=2, strides=2, padding='VALID'))
    return hk.Sequential(layers)


class ConvBlock(hk.Module):
    def __init__(self, out_channels, pool=False):
        super().__init__()
        self.out_channels = out_channels
        self.pool = pool
    
    def __call__(self, x_batch, is_training):
        x = hk.Conv2D(self.out_channels, kernel_shape=3,
                      padding='SAME', with_bias=False)(x_batch)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(x)
        if self.pool:
            x = hk.max_pool(x, window_shape=2, strides=2, padding='VALID')
        return x


class ResNet9(hk.Module):
    def __init__(self, num_classes=10):
        super().__init__(name='ResNet9')
        self.conv_1 = ConvBlock(64)
        self.conv_2 = ConvBlock(128, pool=True)
        self.res_1 = [ConvBlock(128), ConvBlock(128)]
        self.conv_3 = ConvBlock(256, pool=True)
        self.conv_4 = ConvBlock(512, pool=True)
        self.res_2 = [ConvBlock(512), ConvBlock(512)]
        self.clr = hk.Linear(num_classes)
    
    def __call__(self, x_batch, is_training):
        x = self.conv_1(x_batch, is_training)
        x = self.conv_2(x, is_training)
        inp = x  # cloning
        for layer in self.res_1:
            x = layer(x, is_training)
        x += inp
        x = self.conv_3(x, is_training)
        x = self.conv_4(x, is_training)
        inp = x  # cloning
        for layer in self.res_2:
            x = layer(x, is_training)
        x += inp
        x = jax.numpy.mean(x, axis=(1, 2))
        x = self.clr(x)
        return x
