from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.nn import softmax, silu

Array = jnp.ndarray
PyTree = Dict[str, Any]


# ============================================================
# Basic parameter helpers
# ============================================================

def glorot_init(key: Array, in_dim: int, out_dim: int) -> Array:
    limit = jnp.sqrt(6.0 / float(in_dim + out_dim))
    return jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)


def zeros_init(shape) -> Array:
    return jnp.zeros(shape, dtype=jnp.float32)


def init_dense(key: Array, in_dim: int, out_dim: int, bias: bool = True, zero: bool = False) -> PyTree:
    W = zeros_init((in_dim, out_dim)) if zero else glorot_init(key, in_dim, out_dim)
    b = zeros_init((out_dim,)) if bias else None
    return {"W": W, "b": b}


def apply_dense(params: PyTree, x: Array) -> Array:
    y = x @ params["W"]
    if params["b"] is not None:
        y = y + params["b"]
    return y


def init_layer_norm(dim: int, eps: float = 1e-6) -> PyTree:
    return {
        "scale": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
        "eps": jnp.array(eps, dtype=jnp.float32),
    }


def apply_layer_norm(params: PyTree, x: Array) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    xhat = (x - mean) / jnp.sqrt(var + params["eps"])
    return xhat * params["scale"] + params["bias"]


def init_mlp(key: Array, widths, zero_last: bool = False) -> PyTree:
    keys = jax.random.split(key, len(widths) - 1)
    layers = []
    for i, (k, din, dout) in enumerate(zip(keys, widths[:-1], widths[1:])):
        layers.append(init_dense(k, din, dout, bias=True, zero=(zero_last and i == len(widths) - 2)))
    return {"layers": layers}


def apply_mlp(params: PyTree, x: Array, activation=silu) -> Array:
    h = x
    for layer in params["layers"][:-1]:
        h = activation(apply_dense(layer, h))
    return apply_dense(params["layers"][-1], h)


# ============================================================
# Shape helpers
# ============================================================

def _to_heads(x: Array, num_heads: int) -> Array:
    """(B, N, D) -> (B, H, N, d)"""
    B, N, D = x.shape
    if D % num_heads != 0:
        raise ValueError(f"D={D} must be divisible by num_heads={num_heads}")
    dh = D // num_heads
    return x.reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)


def _from_heads(x: Array) -> Array:
    """(B, H, N, d) -> (B, N, D)"""
    B, H, N, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, N, H * dh)


# ============================================================
# Timestep embedding + AdaLN-Zero style modulation
# ============================================================

def timestep_embedding(t: Array, dim: int, max_period: int = 10000) -> Array:
    """
    Standard sinusoidal embedding.
    t: shape (B,) or scalar. Expected in [0, 1] or any real scale.
    returns: (B, dim)
    """
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / max(half, 1))
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def init_time_conditioner(key: Array, dim: int, hidden_dim: Optional[int] = None) -> PyTree:
    """
    Produces per-branch AdaLN parameters and per-feature metric scalings.
    Final projections are zero-initialized so the block starts near identity.
    """
    if hidden_dim is None:
        hidden_dim = 4 * dim
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "embed_mlp": init_mlp(k1, [hidden_dim, hidden_dim, hidden_dim], zero_last=False),
        # shift, scale, gate, log_gamma for token diffusion
        "attn_out": init_dense(k2, hidden_dim, 4 * dim, bias=True, zero=True),
        # shift, scale, gate, log_gamma for channel diffusion
        "ch_out": init_dense(k3, hidden_dim, 4 * dim, bias=True, zero=True),
    }


def apply_time_conditioner(params: PyTree, t: Array, dim: int) -> PyTree:
    emb = timestep_embedding(t, params["embed_mlp"]["layers"][0]["W"].shape[0])
    h = apply_mlp(params["embed_mlp"], emb, activation=silu)
    attn = apply_dense(params["attn_out"], h)
    ch = apply_dense(params["ch_out"], h)

    def split4(y):
        a, b, c, d = jnp.split(y, 4, axis=-1)
        return {"shift": a, "scale": b, "gate": c, "log_gamma": d}

    return {"attn": split4(attn), "ch": split4(ch)}


def adaln_modulate(x_norm: Array, mod: PyTree) -> Array:
    shift = mod["shift"][:, None, :]
    scale = mod["scale"][:, None, :]
    return x_norm * (1.0 + scale) + shift


# ============================================================
# MHDM core: DMAP-style tied Q=K with optional Doob tilt
# ============================================================

def init_mhdm_core(key: Array, dim: int, num_heads: int, bias: bool = False, use_doob: bool = True) -> PyTree:
    if dim % num_heads != 0:
        raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
    keys = jax.random.split(key, 4 if use_doob else 3)
    params = {
        "dim": dim,
        "num_heads": num_heads,
        "qk_proj": init_dense(keys[0], dim, dim, bias=bias),
        "v_proj": init_dense(keys[1], dim, dim, bias=bias),
        "out_proj": init_dense(keys[2], dim, dim, bias=bias),
        "use_doob": use_doob,
    }
    if use_doob:
        # Zero-init => h=1 at initialization.
        params["doob_proj"] = init_dense(keys[3], dim, 1, bias=True, zero=True)
    return params


def _pairwise_neg_sqdist(q: Array, k: Array) -> Array:
    """
    q: (B, H, M, d)
    k: (B, H, N, d)
    returns logits: (B, H, M, N) = -||q-k||^2
    """
    dot = jnp.einsum("bhmd,bhnd->bhmn", q, k)
    q2 = jnp.sum(q * q, axis=-1, keepdims=True)          # (B, H, M, 1)
    k2 = jnp.sum(k * k, axis=-1)[:, :, None, :]          # (B, H, 1, N)
    return 2.0 * dot - q2 - k2


def mhdm_cross_apply(
    params: PyTree,
    query_x: Array,
    support_x: Array,
    query_mod: Optional[PyTree] = None,
    support_mod: Optional[PyTree] = None,
    add_doob: bool = True,
) -> Tuple[Array, PyTree]:
    """
    Cross-operator version for novel-point queries.

    query_x   : (B, M, D)
    support_x : (B, N, D)

    The Doob tilt is added on the destination/support index j:
        softmax_j(logits_ij + phi_j)
    not on the source/query index i.
    """
    num_heads = params["num_heads"]
    dim = params["dim"]

    q_in = query_x if query_mod is None else adaln_modulate(query_x, query_mod)
    s_in = support_x if support_mod is None else adaln_modulate(support_x, support_mod)

    qk_q = apply_dense(params["qk_proj"], q_in)  # (B, M, D)
    qk_s = apply_dense(params["qk_proj"], s_in)  # tied Q=K projection
    v_s = apply_dense(params["v_proj"], s_in)    # (B, N, D)

    # Optional positive feature-wise metric scaling gamma(t).
    if query_mod is not None and "log_gamma" in query_mod:
        qk_q = qk_q * jnp.exp(query_mod["log_gamma"][:, None, :])
    if support_mod is not None and "log_gamma" in support_mod:
        qk_s = qk_s * jnp.exp(support_mod["log_gamma"][:, None, :])

    qh = _to_heads(qk_q, num_heads)  # (B, H, M, d)
    kh = _to_heads(qk_s, num_heads)  # (B, H, N, d)
    vh = _to_heads(v_s, num_heads)   # (B, H, N, d)

    logits = _pairwise_neg_sqdist(qh, kh)
    logits = logits / jnp.sqrt(float(dim // num_heads))

    phi = None
    if params.get("use_doob", False) and add_doob:
        phi = apply_dense(params["doob_proj"], s_in)[..., 0]   # (B, N)
        logits = logits + phi[:, None, None, :]  # destination-side potential

    attn = softmax(logits, axis=-1)
    w = jnp.einsum("bhmn,bhnd->bhmd", attn, vh)
    out = apply_dense(params["out_proj"], _from_heads(w))

    return out, {"attn": attn, "logits": logits, "phi": phi}


def mhdm_self_apply(
    params: PyTree,
    x: Array,
    mod: Optional[PyTree] = None,
    add_doob: bool = True,
) -> Tuple[Array, PyTree]:
    return mhdm_cross_apply(
        params=params,
        query_x=x,
        support_x=x,
        query_mod=mod,
        support_mod=mod,
        add_doob=add_doob,
    )


# ============================================================
# Channel diffusion core
# ============================================================

def init_channel_diffusion_core(
    key: Array,
    dim: int,
    num_heads: int = 8,
    bias: bool = False,
    learn_temperature: bool = True,
) -> PyTree:
    if dim % num_heads != 0:
        raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "dim": dim,
        "num_heads": num_heads,
        "dh": dim // num_heads,
        "qk_proj": init_dense(k1, dim, dim, bias=bias),
        "v_proj": init_dense(k2, dim, dim, bias=bias),
        "out_proj": init_dense(k3, dim, dim, bias=bias),
    }
    if learn_temperature:
        params["tau"] = jnp.ones((num_heads, 1, 1), dtype=jnp.float32)
    return params


def channel_diffusion_apply(params: PyTree, x: Array, mod: Optional[PyTree] = None) -> Tuple[Array, PyTree]:
    """
    Diffusion across channel-vectors, mirroring the PyTorch block.
    x: (B, N, D)
    """
    B, N, D = x.shape
    H = params["num_heads"]
    dh = params["dh"]

    x_in = x if mod is None else adaln_modulate(x, mod)

    qk = apply_dense(params["qk_proj"], x_in)  # (B, N, D)
    v = apply_dense(params["v_proj"], x_in)    # (B, N, D)

    if mod is not None and "log_gamma" in mod:
        qk = qk * jnp.exp(mod["log_gamma"][:, None, :])

    # Channels-as-sequence: (B, D, N)
    qk = qk.transpose(0, 2, 1)
    v = v.transpose(0, 2, 1)

    # Split across channels: (B, H, dh, N)
    qk = qk.reshape(B, H, dh, N)
    v = v.reshape(B, H, dh, N)

    dot = jnp.einsum("bhcn,bhdn->bhcd", qk, qk)
    q2 = jnp.sum(qk * qk, axis=-1)
    logits = 2.0 * dot - q2[..., :, None] - q2[..., None, :]
    logits = logits / jnp.sqrt(float(max(N, 1)))

    if "tau" in params:
        logits = logits * params["tau"][None, ...]

    attn = softmax(logits, axis=-1)
    w = jnp.einsum("bhcd,bhdn->bhcn", attn, v)

    out = w.reshape(B, D, N).transpose(0, 2, 1)
    out = apply_dense(params["out_proj"], out)
    return out, {"attn": attn, "logits": logits}


# ============================================================
# Full diffusion block with time conditioning
# ============================================================

def init_diffusion_block(
    key: Array,
    dim: int,
    num_heads: int,
    channel_heads: int = 8,
    bias: bool = False,
    use_doob: bool = True,
) -> PyTree:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "norm1": init_layer_norm(dim),
        "attn": init_mhdm_core(k1, dim=dim, num_heads=num_heads, bias=bias, use_doob=use_doob),
        "norm2": init_layer_norm(dim),
        "channel": init_channel_diffusion_core(k2, dim=dim, num_heads=channel_heads, bias=bias),
        "time": init_time_conditioner(k3, dim=dim),
        # LayerScale / AdaLN-Zero style residual gates start tiny.
        "gamma1": 1e-4 * jnp.ones((dim,), dtype=jnp.float32),
        "gamma2": 1e-4 * jnp.ones((dim,), dtype=jnp.float32),
    }


def diffusion_block_apply(
    params: PyTree,
    x: Array,
    t: Array,
    *,
    add_doob: bool = True,
) -> Tuple[Array, PyTree]:
    """
    x: (B, N, D)
    t: scalar or (B,) timestep / noise level
    """
    D = x.shape[-1]
    mods = apply_time_conditioner(params["time"], t, dim=D)

    # Token diffusion branch
    x1 = apply_layer_norm(params["norm1"], x)
    attn_out, attn_aux = mhdm_self_apply(
        params["attn"],
        x=x1,
        mod=mods["attn"],
        add_doob=add_doob,
    )
    gate1 = mods["attn"]["gate"][:, None, :]
    x = x + attn_out * gate1 * params["gamma1"]

    # Channel diffusion branch
    x2 = apply_layer_norm(params["norm2"], x)
    ch_out, ch_aux = channel_diffusion_apply(params["channel"], x2, mod=mods["ch"])
    gate2 = mods["ch"]["gate"][:, None, :]
    x = x + ch_out * gate2 * params["gamma2"]

    aux = {
        "mods": mods,
        "attn": attn_aux,
        "channel": ch_aux,
    }
    return x, aux


# ============================================================
# Example usage
# ============================================================

def _demo() -> None:
    key = jax.random.PRNGKey(0)
    B, N, D = 2, 16, 32
    H = 4

    x = jax.random.normal(key, (B, N, D))
    t = jnp.array([0.25, 0.75], dtype=jnp.float32)

    block = init_diffusion_block(key, dim=D, num_heads=H, channel_heads=H, use_doob=True)
    y, aux = diffusion_block_apply(block, x, t)
    print("input shape ", x.shape)
    print("output shape", y.shape)
    print("token attn  ", aux["attn"]["attn"].shape)
    print("channel attn", aux["channel"]["attn"].shape)

    # Novel-point query against a support cloud using the same learned kernel.
    query = jax.random.normal(jax.random.PRNGKey(1), (B, 1, D))
    x_norm = apply_layer_norm(block["norm1"], x)
    q_norm = apply_layer_norm(block["norm1"], query)
    mods = apply_time_conditioner(block["time"], t, dim=D)
    qy, _ = mhdm_cross_apply(
        block["attn"],
        query_x=q_norm,
        support_x=x_norm,
        query_mod=mods["attn"],
        support_mod=mods["attn"],
        add_doob=True,
    )
    print("query output", qy.shape)
