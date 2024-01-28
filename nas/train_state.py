import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from clu import metrics
import optax
from clu import metrics
from flax import struct, core
from typing import Callable, Any


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class NasTrainState(struct.PyTreeNode):
    rng: jax.Array
    metrics: Metrics
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    w_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    h_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    bn_state: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    inner_opt: optax.GradientTransformation = struct.field(pytree_node=False)
    inner_opt_state: optax.OptState = struct.field(pytree_node=True)
    outer_opt: optax.GradientTransformation = struct.field(pytree_node=False)
    outer_opt_state: optax.OptState = struct.field(pytree_node=True)
    scheduler: Callable = struct.field(pytree_node=False)


    @classmethod
    def create(cls, *, apply_fn, w_params, h_params, bn_state, inner_opt, scheduler, outer_opt, **kwargs):
        inner_opt_state = inner_opt.init(w_params)
        outer_opt_state = outer_opt.init(h_params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            w_params=w_params,
            h_params=h_params,
            bn_state=bn_state,
            inner_opt=inner_opt,
            outer_opt=outer_opt,
            inner_opt_state=inner_opt_state,
            outer_opt_state=outer_opt_state,
            scheduler=scheduler,
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
        

def create_nas_train_state(module, rng, total_steps, learning_rate=0.025, momentum=0.9, w_decay=3e-4,
                               alpha_lr=1e-4, alpha_decay=1e-3):
    params, bn_state = module.init(rng, jnp.zeros([1, 32, 32, 3]), True)
    w_params, h_params = hk.data_structures.partition(lambda m, n, p: 'alpha' not in n, params)
    sch = optax.cosine_decay_schedule(learning_rate, total_steps)
    tx_inner = optax.chain(optax.add_decayed_weights(w_decay),
                           optax.sgd(sch, momentum=momentum))
    tx_outer = optax.chain(optax.add_decayed_weights(alpha_decay), optax.adam(alpha_lr, b1=0.5, b2=0.999))
    return NasTrainState.create(
      apply_fn=module.apply, w_params=w_params, h_params=h_params, bn_state=bn_state,
      inner_opt=tx_inner, outer_opt=tx_outer, scheduler=sch,
      metrics=Metrics.empty(), rng=rng)



# conv_net = hk.transform_with_state(lambda x, t: CNN(16, 10, 1, drop_path_prob=0.1)(x, t))
# rng = jax.random.PRNGKey(42)
# state = create_nas_train_state(conv_net, jax.random.PRNGKey(0))
