import jax
import jax.numpy as jnp
from flax import struct, core
from flax.training import train_state
import optax
import optax
from clu import metrics
from typing import Callable, Any


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class DataCleanTrainState(train_state.TrainState):
    rng: jax.Array
    metrics: Metrics
    scheduler: Callable
    decay: float
    bn_state: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    @property
    def lr(self):
        return self.scheduler(self.step)


def create_train_state(module, rng, inner_steps, learning_rate=0.1, momentum=0.9, decay=3e-4):
    """Creates an initial `TrainState`."""
    params, bn_state = module.init(rng, jnp.ones([1, 32, 32, 3]), True)
    sch = optax.cosine_decay_schedule(learning_rate, inner_steps)
    opt = optax.chain(optax.add_decayed_weights(decay),
                      optax.sgd(learning_rate, momentum))
    return DataCleanTrainState.create(apply_fn=module.apply,
                                      rng=rng,
                                      params=params, bn_state=bn_state,
                                      tx=opt, metrics=Metrics.empty(), scheduler=sch, decay=decay)
