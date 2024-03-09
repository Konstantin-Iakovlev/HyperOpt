import jax
import jax.numpy as jnp
import haiku as hk
import optax
from train_state import BiLevelTrainState


@jax.jit
def loss_fn(w_params, h_params, state, batch):
    params = hk.data_structures.merge(w_params, h_params)
    logits = state.apply_fn(params, None, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    return loss


@jax.jit
def compute_metrics(*, state, batch):
    params = hk.data_structures.merge(state.w_params, state.h_params)
    logits = state.apply_fn(params, None, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def inner_step(state: BiLevelTrainState, batch):    
    grad_fn = jax.grad(loss_fn, argnums=0)
    grads = grad_fn(state.w_params, state.h_params, state, batch)
    state = state.apply_w_gradients(w_grads=grads)
    return state


@jax.jit
def normalize(v):
    return jnp.sqrt(jax.tree_util.tree_reduce(lambda v, x: v + (x ** 2).sum(), v, 0))


@jax.jit
def B_jvp(w_params, h_params, batch, state, v, r=1e-2):
    """d^2 L1 / dl dw v"""
    eps = r / normalize(v)
    w_plus = jax.tree_util.tree_map(lambda x, y: x + eps * y, w_params, v)
    w_minus = jax.tree_util.tree_map(lambda x, y: x - eps * y, w_params, v)
    dl_dlam = jax.grad(loss_fn, argnums=1)
    g_plus = dl_dlam(w_plus, h_params, state, batch)
    g_minus = dl_dlam(w_minus, h_params, state, batch)
    return jax.tree_util.tree_map(lambda x, y: -state.lr * (x - y) / (2 * eps), g_plus, g_minus)


@jax.jit
def A_jvp(w_params, batch, state, v, r=1e-2):
    eps = r / normalize(v)
    w_plus = jax.tree_util.tree_map(lambda x, y: x + eps * y, w_params, v)
    w_minus = jax.tree_util.tree_map(lambda x, y: x - eps * y, w_params, v)
    dl_dw = jax.grad(loss_fn, argnums=0)
    g_plus = dl_dw(w_plus, state.h_params, state, batch)
    g_minus = dl_dw(w_minus, state.h_params, state, batch)
    hvp = jax.tree_util.tree_map(lambda x, y: (x - y) / (2 * eps), g_plus, g_minus)
    return jax.tree_util.tree_map(lambda x, y: x - state.lr * y, v, hvp)


@jax.jit
def fo_grad(state, val_batch):
    return jax.grad(loss_fn, argnums=1)(state.w_params, state.h_params, state, val_batch)


def drmad_grad(state, batches, val_batch):
    """T = len(batches)"""
    T = len(batches)
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    w_0 = state.w_params
    for step, batch in enumerate(batches):
        state = inner_step(state, batch)
    w_T = state.w_params
    alpha = jax.grad(loss_fn, argnums=0)(state.w_params, state.h_params, state, val_batch)
    for step, batch in enumerate(batches[::-1]):
        t = T - step
        w_tm1 = jax.tree_util.tree_map(lambda x, y: (1 - (t - 1) / T) * x + (t - 1) / T * y, w_0, w_T)
        g_so = jax.tree_util.tree_map(lambda x, y: x + y, B_jvp(w_tm1, state.h_params, batch, state, alpha), g_so)
        # update alpha
        alpha = A_jvp(w_tm1, batch, state, alpha)
    return state, g_so


def proposed_so_grad(state, batches, val_batch, gamma):
    """T = len(batches)"""
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        curr_alpha = jax.grad(loss_fn, argnums=0)(new_state.w_params, state.h_params, state, val_batch)
        g_so = jax.tree_util.tree_map(lambda x, y: x * gamma ** (T - 1 - step) + y,
                                      B_jvp(state.w_params, state.h_params, batch,
                                            state, curr_alpha),
                                     g_so)
        state = new_state
    return state, g_so


def luketina_so_grad(state: BiLevelTrainState, batches, val_batch):
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        if step == len(batches) - 1:
            curr_alpha = jax.grad(loss_fn, argnums=0)(new_state.w_params, state.h_params, state, val_batch)
            g_so = B_jvp(state.w_params, state.h_params, batch, state, curr_alpha)
        state = new_state
    return state, g_so


def IFT_grad(state: BiLevelTrainState, batches, val_batch, N):
    """N + 1 - the number of terms from Neuman series. See (9) from i-DARTS; the number of online opt. steps"""
    for step, batch in enumerate(batches):
        state = inner_step(state, batch)
    v = jax.grad(loss_fn, argnums=0)(state.w_params, state.h_params, state, val_batch)
    so_grad = B_jvp(state.w_params, state.h_params, batches[-1], state, v)
    for k in range(1, N + 1):
        v = A_jvp(state.w_params, batches[-1], state, v)
        hvp = B_jvp(state.w_params, state.h_params, batches[-1], state, v)
        so_grad = jax.tree_util.tree_map(
            lambda x, y: x + y, so_grad, hvp)
    return state, so_grad
