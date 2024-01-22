import jax
import jax.numpy as jnp
from train_state import DataWTrainState
import optax
import haiku as hk


@jax.jit
def loss_fn_trn(w_params, h_params, state: DataWTrainState, batch, is_training=True):
    noize = jax.random.normal(state.rng, shape=(*batch['image'].shape[:-1], 1))
    inp_image = jnp.concatenate(batch['image'], noize, axis=-1)
    aug_img = state.wnet_fn(h_params, state.rng, inp_image)[..., :-1]
    logits, bn_state = state.apply_fn(w_params, state.bn_state, state.rng,
                                          aug_img,
                                          is_training)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label'])
    return loss.mean(), state.replace(bn_state=bn_state)


@jax.jit
def loss_fn_val(w_params, h_params, state, batch, is_training=True):
    logits, bn_state = state.apply_fn(w_params, state.bn_state, state.rng,
                                          batch['image'], is_training)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    return loss, state.replace(bn_state=bn_state)


@jax.jit
def loss_fn_val_grad_params(w_params, h_params, state, batch):
    # is_training is always True
    return jax.grad(loss_fn_val, argnums=0, has_aux=True)(w_params, h_params, state, batch)[0]


@jax.jit
def loss_fn_trn_grad_params(w_params, h_params, state, batch):
    return jax.grad(loss_fn_trn, argnums=0, has_aux=True)(w_params, h_params, state, batch)[0]


@jax.jit
def compute_metrics(*, state, batch):
    params = hk.data_structures.merge(state.w_params, state.h_params)
    logits, _ = state.apply_fn(params, state.bn_state, None, batch['image'], False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def inner_step(state: DataWTrainState, batch):    
    grad_fn = jax.value_and_grad(loss_fn_trn, argnums=0, has_aux=True)
    (_, state), grads = grad_fn(state.w_params, state.h_params, state, batch)
    state = state.apply_w_gradients(w_grads=grads)
    return state


@jax.jit
def inner_step_baseline(state: DataWTrainState, batch):
    grad_fn = jax.value_and_grad(loss_fn_val, argnums=0, has_aux=True)
    (_, state), grads = grad_fn(state.w_params, state.h_params, state, batch)
    state = state.apply_w_gradients(w_grads=grads)
    return state


@jax.jit
def B_jvp(w_params, h_params, batch, state, v, eps=1e-7):
    """d^2 L1 / dl dw v"""
    w_plus = jax.tree_util.tree_map(lambda x, y: x + eps * y, w_params, v)
    w_minus = jax.tree_util.tree_map(lambda x, y: x - eps * y, w_params, v)
    dl_dlam = jax.grad(loss_fn_trn, argnums=1, has_aux=True)
    g_plus = dl_dlam(w_plus, h_params, state, batch)[0]
    g_minus = dl_dlam(w_minus, h_params, state, batch)[0]
    # print(g_plus)
    return jax.tree_util.tree_map(lambda x, y: -state.lr * (x - y) / (2 * eps), g_plus, g_minus)


@jax.jit
def A_jvp(w_params, batch, state, v, eps=1e-7):
    w_plus = jax.tree_util.tree_map(lambda x, y: x + eps * y, w_params, v)
    w_minus = jax.tree_util.tree_map(lambda x, y: x - eps * y, w_params, v)
    dl_dw = jax.grad(loss_fn_trn, argnums=0, has_aux=True)
    g_plus = dl_dw(w_plus, state.h_params, state, batch)[0]
    g_minus = dl_dw(w_minus, state.h_params, state, batch)[0]
    hvp = jax.tree_util.tree_map(lambda x, y: (x - y) / (2 * eps), g_plus, g_minus)
    return jax.tree_util.tree_map(lambda x, y: x - state.lr * y, v, hvp)


def drmad_grad(state, batches, val_batch):
    """T = len(batches)"""
    T = len(batches)
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    w_0 = state.w_params
    for step, batch in enumerate(batches):
        state = inner_step(state, batch)
    w_T = state.w_params
    alpha = loss_fn_val_grad_params(state.w_params, state.h_params, state, val_batch)
    for step, batch in enumerate(batches[::-1]):
        t = T - step
        w_tm1 = jax.tree_util.tree_map(lambda x, y: (1 - (t - 1) / T) * x + (t - 1) / T * y, w_0, w_T)
        g_so = jax.tree_util.tree_map(lambda x, y: x + y, B_jvp(w_tm1, state.h_params, batch, state, alpha), g_so)
        # update alpha
        if step != len(batches) - 1:
            alpha = A_jvp(w_tm1, batch, state, alpha)
    return state, g_so


def proposed_so_grad(state, batches, val_batch, gamma):
    """T = len(batches)"""
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        curr_alpha = loss_fn_val_grad_params(new_state.w_params, state.h_params, state, val_batch)
        g_so = jax.tree_util.tree_map(lambda x, y: x * gamma ** (T - 1 - step) + y,
                                      B_jvp(state.w_params, state.h_params, batch,
                                            state, curr_alpha),
                                     g_so)
        state = new_state
    return state, g_so


def fish_so_grad(state, batches, val_batch):
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        curr_alpha = loss_fn_val_grad_params(new_state.w_params, state.h_params, state, val_batch)
        v = loss_fn_trn_grad_params(new_state.w_params, state.h_params, state, batch)
        B_alpha_jvp = B_jvp(state.w_params, state.h_params, batch, state, curr_alpha)
        B_v_jvp = B_jvp(state.w_params, state.h_params, batch, state, v)

        v_norm_sq = jax.tree_util.tree_reduce(lambda s, x: s + (x ** 2).sum(), v, initializer=0.0)
        c_t = ((1 + state.lr * v_norm_sq) ** (T - 1 - step) - 1) / v_norm_sq  # TODO: expm1 is more stable operation
        alpha_dot_v = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: (x * y).sum(), curr_alpha, v)))
        g_so_upd = jax.tree_util.tree_map(lambda x, y: x - c_t * alpha_dot_v * y, B_alpha_jvp, B_v_jvp)
        g_so = jax.tree_util.tree_map(lambda x, y: x + y, g_so, g_so_upd)
        state = new_state
    return state, g_so


def luketina_so_grad(state, batches, val_batch):
    """T = len(batches)"""
    g_so = jax.tree_util.tree_map(jnp.zeros_like, state.h_params)
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        if step == T - 1:
            curr_alpha = loss_fn_val_grad_params(new_state.w_params, state.h_params, state, val_batch)
            g_so = jax.tree_util.tree_map(lambda x, y: x + y,
                                        B_jvp(state.w_params, state.h_params, batch,
                                                state, curr_alpha), g_so)
        state = new_state
    return state, g_so


def IFT_grad(state, batches, val_batch, N):
    """N - the number of terms from Neuman series"""
    for step, batch in enumerate(batches):
        state = inner_step(state, batch)
        
    v = loss_fn_val_grad_params(state.w_params, state.h_params, state, val_batch)
    so_grad = B_jvp(state.w_params, state.h_params, batches[-1], state, v)
    for k in range(1, N + 1):
        v = A_jvp(state.w_params, batches[-1], state, v)
        hvp = B_jvp(state.w_params, state.h_params, batches[-1], state, v)
        so_grad = jax.tree_util.tree_map(lambda x, y: x + y, so_grad, hvp)
    return state, so_grad
