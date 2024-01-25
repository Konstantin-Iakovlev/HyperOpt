import jax
import jax.numpy as jnp
from flax import struct, core
import optax
import optax
from clu import metrics
from typing import Callable, Any


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class DataWTrainState(struct.PyTreeNode):
    rng: jax.Array
    metrics: Metrics
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    wnet_fn: Callable = struct.field(pytree_node=False)
    w_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    h_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    bn_state: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    inner_opt: optax.GradientTransformation = struct.field(pytree_node=False)
    scheduler: Callable = struct.field(pytree_node=False)
    inner_opt_state: optax.OptState = struct.field(pytree_node=True)
    outer_opt: optax.GradientTransformation = struct.field(pytree_node=False)
    outer_opt_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, wnet_fn, w_params, h_params,
               bn_state, inner_opt, scheduler, outer_opt, **kwargs):
        inner_opt_state = inner_opt.init(w_params)
        outer_opt_state = outer_opt.init(h_params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            wnet_fn=wnet_fn,
            w_params=w_params,
            h_params=h_params,
            bn_state=bn_state,
            inner_opt=inner_opt,
            scheduler=scheduler,
            outer_opt=outer_opt,
            inner_opt_state=inner_opt_state,
            outer_opt_state=outer_opt_state,
            **kwargs,
        )

    def apply_w_gradients(self, *, w_grads, **kwargs):
        updates, new_inn_state = self.inner_opt.update(w_grads, self.inner_opt_state, self.w_params)
        new_params = optax.apply_updates(self.w_params, updates)
        rng, _ = jax.random.split(self.rng)
        return self.replace(
            step=self.step + 1,
            w_params=new_params,
            inner_opt_state=new_inn_state,
            rng=rng,
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
    
    @property
    def lr(self):
        return self.scheduler(self.step)
        

def create_dw_train_state(module, wnet, rng, inner_steps, learning_rate=0.025, momentum=0.9, w_decay=3e-4,
                               alpha_lr=1e-4, alpha_decay=1e-4, input_shape=[28, 28, 1]):
    """Creates an initial `TrainState`."""
    w_params, bn_state = module.init(rng, jnp.ones([1] + input_shape), True)
    h_params = wnet.init(rng, jnp.ones([1, 32, 32, 3]), rng)
    sch = optax.cosine_decay_schedule(learning_rate, inner_steps)
    tx_inner = optax.chain(optax.add_decayed_weights(w_decay),
                           optax.sgd(sch, momentum=momentum))
    tx_outer = optax.chain(optax.add_decayed_weights(alpha_decay),
                           optax.adam(alpha_lr, b1=0.5, b2=0.999))
    return DataWTrainState.create(
      apply_fn=module.apply, wnet_fn=wnet.apply,
        w_params=w_params, h_params=h_params,
        bn_state=bn_state, inner_opt=tx_inner, scheduler=sch,
        outer_opt=tx_outer,
      metrics=Metrics.empty(), rng=rng)
