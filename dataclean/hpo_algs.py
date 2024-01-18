import jax
import optax
from train_state import DataCleanTrainState


@jax.jit
def loss_fn(params, state: DataCleanTrainState, batch, is_training=True):
    logits, bn_state = state.apply_fn(params, state.bn_state, state.rng,
                                      batch['image'], is_training)
    loss = (optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']) * jax.nn.sigmoid(batch['lambda'])).mean()
    return loss, state.replace(bn_state=bn_state)


@jax.jit
def compute_metrics(*, state, batch):
    logits, _ = state.apply_fn(state.params, state.bn_state, None, batch['image'], False)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def inner_step(state: DataCleanTrainState, batch):
    grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
    grads = grad_fn(state.params, state, batch)[0]
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def inner_step_baseline(state: DataCleanTrainState, batch):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (_, state), grads = grad_fn(state.params, state, batch)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def B_jvp(params, batch, state, v):
    w_plus = jax.tree_util.tree_map(lambda x, y: x + 1e-7 * y, params, v)
    w_minus = jax.tree_util.tree_map(lambda x, y: x - 1e-7 * y, params, v)

    def fun(w, h):
        return loss_fn(w, state,
                       {'image': batch['image'], 'label': batch['label'], 'lambda': h})
    dl_dlam = jax.grad(fun, argnums=1, has_aux=True)
    g_plus = dl_dlam(w_plus, batch['lambda'])[0]
    g_minus = dl_dlam(w_minus, batch['lambda'])[0]
    return -state.lr * (g_plus - g_minus) / (2 * 1e-7)


@jax.jit
def A_jvp(params, batch, state, v):
    w_plus = jax.tree_util.tree_map(lambda x, y: x + 1e-7 * y, params, v)
    w_minus = jax.tree_util.tree_map(lambda x, y: x - 1e-7 * y, params, v)
    dl_dw = jax.grad(loss_fn, argnums=0, has_aux=True)
    g_plus = dl_dw(w_plus, state, batch)[0]
    g_minus = dl_dw(w_minus, state, batch)[0]
    hvp = jax.tree_util.tree_map(lambda x, y: (
        x - y) / (2 * 1e-7), g_plus, g_minus)
    return jax.tree_util.tree_map(lambda x, y: x - state.lr * y, v, hvp)


def proposed_so_grad(state, batches, val_batch, gamma):
    """T = len(batches)"""
    g_so_arr = []
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        curr_alpha = jax.grad(loss_fn, argnums=0, has_aux=True)(new_state.params, state, val_batch)[0]
        g_so_arr.append(B_jvp(state.params, batch, state,
                        curr_alpha) * gamma ** (T - 1 - step))
        state = new_state
    return state, g_so_arr


def fish_so_grad(state: DataCleanTrainState, batches, val_batch):
    """T = len(batches)"""
    g_so_arr = []
    T = len(batches)
    for step, batch in enumerate(batches):
        new_state = inner_step(state, batch)
        curr_alpha = jax.grad(loss_fn, argnums=0, has_aux=True)(new_state.params, state, val_batch)[0]
        v = jax.grad(loss_fn, argnums=0, has_aux=True)(new_state.params, state, batch)[0]
        B_alpha_jvp = B_jvp(state.params, batch, state, curr_alpha)
        B_v_jvp = B_jvp(state.params, batch, state, v)

        v_norm_sq = jax.tree_util.tree_reduce(
            lambda s, x: s + (x ** 2).sum(), v, initializer=0.0)
        c_t = ((1 + state.lr * v_norm_sq) ** (T - 1 - step) - 1) / \
            v_norm_sq  # TODO: expm1 is more stable operation
        # c_t = state.lr * (T - 1 - step)  # Taylor approximation
        alpha_dot_v = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: (x * y).sum(),
                                                                           curr_alpha, v)))
        g_so = jax.tree_util.tree_map(
            lambda x, y: x - c_t * alpha_dot_v * y, B_alpha_jvp, B_v_jvp)
        g_so_arr.append(g_so)
        state = new_state

    return state, g_so_arr


def drmad_grad(state, batches, val_batch):
    """T = len(batches)"""
    g_so_arr = []
    T = len(batches)
    w_0 = state.params
    for step, batch in enumerate(batches):
        state = inner_step(state, batch)
    w_T = state.params
    alpha = jax.grad(loss_fn, argnums=0, has_aux=True)(state.params, state, val_batch)[0]
    for step, batch in enumerate(batches[::-1]):
        t = T - step
        w_tm1 = jax.tree_util.tree_map(lambda x, y: (
            1 - (t - 1) / T) * x + (t - 1) / T * y, w_0, w_T)
        g_so_arr.append(B_jvp(w_tm1, batch, state, alpha))
        # update alpha
        alpha = A_jvp(w_tm1, batch, state, alpha)
    return state, g_so_arr


def IFT_grad(state, batches, val_batch, N, K):
    """N + 1 - the number of terms from Neuman series. See (9) from i-DARTS; the number of online opt. steps"""
    g_so_arr = []
    assert len(batches) % K == 0
    for step, batch in enumerate(batches):
        state = inner_step(state, batch)
        if step >= len(batches) - K:
            v = jax.grad(loss_fn, argnums=0, has_aux=True)(state.params, state, val_batch)[0]
            so_grad = B_jvp(state.params, batches[-1], state, v)
            for k in range(1, N + 1):
                v = A_jvp(state.params, batches[-1], state, v)
                hvp = B_jvp(state.params, batches[-1], state, v)
                so_grad = jax.tree_util.tree_map(
                    lambda x, y: x + y, so_grad, hvp)
            g_so_arr.append(so_grad)
            # (N + 1) * K JVPs
    return state, g_so_arr * (len(batches) // K)
