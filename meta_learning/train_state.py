import jax
import jax.numpy as jnp
from flax import struct, core
from flax.training import train_state
import haiku as hk
import optax
import optax
from clu import metrics
from typing import Callable, Any


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class BiLevelTrainState(struct.PyTreeNode):
    metrics: Metrics
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    w_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    h_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    inner_opt: optax.GradientTransformation = struct.field(pytree_node=False)
    inner_opt_state: optax.OptState = struct.field(pytree_node=True)
    outer_opt: optax.GradientTransformation = struct.field(pytree_node=False)
    outer_opt_state: optax.OptState = struct.field(pytree_node=True)
    lr: float  # inner lr


    @classmethod
    def create(cls, *, apply_fn, w_params, h_params, inner_opt, outer_opt, **kwargs):
        inner_opt_state = inner_opt.init(w_params)
        outer_opt_state = outer_opt.init(h_params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            w_params=w_params,
            h_params=h_params,
            inner_opt=inner_opt,
            outer_opt=outer_opt,
            inner_opt_state=inner_opt_state,
            outer_opt_state=outer_opt_state,
            **kwargs,
        )

    def apply_w_gradients(self, *, w_grads, **kwargs):
        updates, new_inn_state = self.inner_opt.update(w_grads, self.inner_opt_state, self.w_params)
        new_params = optax.apply_updates(self.w_params, updates)
        return self.replace(
            step=self.step + 1,
            w_params=new_params,
            inner_opt_state=new_inn_state,
            **kwargs
        )

    def apply_h_gradients(self, *, h_grads, **kwargs):
        updates, new_out_state = self.outer_opt.update(h_grads, self.outer_opt_state, self.h_params)
        new_params = optax.apply_updates(self.h_params, updates)
        return self.replace(
            step=self.step + 1,
            h_params=new_params,
            outer_opt_state=new_out_state,
            **kwargs
        )
        

def create_bilevel_train_state(module, rng, learning_rate, outer_lr, momentum=0.9, weight_decay=1e-4):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, 32, 32, 3]))
    w_params, h_params = hk.data_structures.partition(lambda m, n, p: 'linear' in m, params)
    tx_inner = optax.chain(optax.add_decayed_weights(weight_decay),
                           optax.sgd(learning_rate, momentum=momentum))
    tx_outer = optax.adam(outer_lr)
    return BiLevelTrainState.create(
      apply_fn=module.apply, w_params=w_params, h_params=h_params, inner_opt=tx_inner, outer_opt=tx_outer,
        metrics=Metrics.empty(), lr=learning_rate)
    