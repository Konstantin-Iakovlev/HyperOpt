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


class DataCleanTrainState(struct.PyTreeNode):
    rng: jax.Array
    metrics: Metrics
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    bn_state: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    inner_opt: optax.GradientTransformation = struct.field(pytree_node=False)
    scheduler: Callable = struct.field(pytree_node=False)
    inner_opt_state: optax.OptState = struct.field(pytree_node=True)
    decay: float

    @classmethod
    def create(cls, *, apply_fn, params,
               bn_state, inner_opt, scheduler, **kwargs):
        inner_opt_state = inner_opt.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            bn_state=bn_state,
            inner_opt=inner_opt,
            scheduler=scheduler,
            inner_opt_state=inner_opt_state,
            **kwargs,
        )

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_inn_state = self.inner_opt.update(grads, self.inner_opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        rng, _ = jax.random.split(self.rng)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            inner_opt_state=new_inn_state,
            rng=rng,
            **kwargs
        )
    
    @property
    def lr(self):
        return self.scheduler(self.step)


def create_train_state(module, rng, inner_steps, learning_rate=0.1, momentum=0.9, decay=3e-4,
                       inp_shape=[32, 32, 3]):
    """Creates an initial `TrainState`."""
    params, bn_state = module.init(rng, jnp.ones([1] + inp_shape), True)
    sch = optax.cosine_decay_schedule(learning_rate, inner_steps)
    opt = optax.chain(optax.add_decayed_weights(decay),
                      optax.sgd(learning_rate, momentum))
    return DataCleanTrainState.create(apply_fn=module.apply,
                                      rng=rng,
                                      params=params, bn_state=bn_state,
                                      inner_opt=opt, metrics=Metrics.empty(), scheduler=sch, decay=decay)
