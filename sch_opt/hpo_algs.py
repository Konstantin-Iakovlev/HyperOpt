import jax
import jax.numpy as jnp
from train_state import SchTrainState
import optax
import haiku as hk


def loss_fn(w_params, state: SchTrainState, batch, is_training=True):
    logits, bn_state = state.apply_fn(w_params, state.bn_state, state.rng,
                                          batch['image'],
                                          is_training)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    return loss, state.replace(bn_state=bn_state)


@jax.jit
def loss_fn_val_grad_params(w_params, state, batch):
    # is_training is always True
    return jax.grad(loss_fn, argnums=0, has_aux=True)(w_params, state, batch, False)[0]


@jax.jit
def loss_fn_trn_grad_params(w_params, state, batch):
    return jax.grad(loss_fn, argnums=0, has_aux=True)(w_params, state, batch, True)[0]


@jax.jit
def compute_metrics(*, state, batch):
    logits, _ = state.apply_fn(state.w_params, state.bn_state, None, batch['image'], False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def inner_step(state: SchTrainState, batch):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (_, state), grads = grad_fn(state.w_params, state, batch, True)
    state = state.apply_w_gradients(w_grads=grads)
    return state


@jax.jit
def B_jvp(w_params, batch, state: SchTrainState, v):
    dl_trn_dw = loss_fn_trn_grad_params(w_params, state, batch)
    grad_fn_h = jax.grad(lambda h: state.lr_schedule(
        state.step, h['alpha_0'], h['beta']))
    eta_grad = grad_fn_h(state.h_params)
    dot_prod = -jax.tree_util.tree_reduce(lambda x, y: x + y,
                                         jax.tree_util.tree_map(
                                             lambda x, y: (x * y).sum(), dl_trn_dw, v),
                                         initializer=0.0)
    return jax.tree_util.tree_map(lambda x: dot_prod * x, eta_grad)


def proposed_so_grad(state, batches, val_batch, gamma):
    """T = len(batches)"""
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        curr_alpha = loss_fn_val_grad_params(new_state.w_params, state, val_batch)
        g_so = jax.tree_util.tree_map(lambda x, y: x * (1 - gamma) ** (T - 1 - step) + y,
                                      B_jvp(state.w_params, batch, state, curr_alpha), g_so)
        state = new_state
    return state, g_so


def luketina_so_grad(state, batches, val_batch):
    """T = len(batches)"""
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        if step == T - 1:
            curr_alpha = loss_fn_val_grad_params(new_state.w_params, state, val_batch)
            g_so = jax.tree_util.tree_map(lambda x, y: x + y,
                                        B_jvp(state.w_params, batch, state, curr_alpha), g_so)
        state = new_state
    return state, g_so
