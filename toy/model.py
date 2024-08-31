import jax
import haiku as hk
import jax.numpy as jnp

S = .1

class Toy(hk.Module):
    def __init__(self):
        super().__init__(name="toy")
        
    def __call__(self, *args):
        x = hk.get_parameter("x", shape=[1], dtype=jnp.float32, init=hk.initializers.RandomNormal(S))
        y = hk.get_parameter("y", shape=[1], dtype=jnp.float32, init=hk.initializers.RandomNormal(S))
        return x, y
        
class Toy2(hk.Module):
    def __init__(self):
        super().__init__(name="toy2")
        
    def __call__(self, *args):
        l = hk.get_parameter("l", shape=[1], dtype=jnp.float32, init=hk.initializers.RandomNormal(S))
        
        return l
