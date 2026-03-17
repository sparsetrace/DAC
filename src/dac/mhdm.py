import math
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax


Array = jnp.ndarray


# ============================================================
# Helpers
# ============================================================

def _to_heads(x: Array, num_heads: int) -> Array:
    """
    (B, N, D) -> (B, H, N, d)
    """
    B, N, D = x.shape
    if D % num_heads != 0:
        raise ValueError(f"D={D} must be divisible by num_heads={num_heads}")
    dh = D // num_heads
    return x.reshape(B, N, num_heads, dh).transpose(0, 2, 1, 3)


def _from_heads(x: Array) -> Array:
    """
    (B, H, N, d) -> (B, N, D)
    """
    B, H, N, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, N, H * dh)


def timestep_embedding(t: Array, dim: int, max_period: int = 10000) -> Array:
    """
    Standard sinusoidal embedding.
    t: scalar or shape (B,)
    returns: (B, dim)
    """
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim == 0:
        t = t[None]

    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / max(half, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def modulate(x: Array, shift: Array, scale: Array) -> Array:
    """
    x:     (B, N, D)
    shift: (B, D)
    scale: (B, D)
    """
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


# ============================================================
# AdaLN-Zero style time conditioner
# ============================================================

class AdaLNTimeConditioner(nn.Module):
    """
    Produces per-branch:
      shift, scale, gate, log_gamma
    where:
      - shift/scale are AdaLN params
      - gate is DiT-style residual gating
      - log_gamma is the time-dependent metric scaling
    """
    dim: int
    hidden_dim: Optional[int] = None

    @nn.compact
    def __call__(self, t: Array) -> Dict[str, Dict[str, Array]]:
        hidden = self.hidden_dim or (4 * self.dim)

        emb = timestep_embedding(t, hidden)
        h = nn.Dense(hidden)(emb)
        h = nn.silu(h)
        h = nn.Dense(hidden)(h)
        h = nn.silu(h)

        # zero init => near-identity block at initialization
        zero_kernel = nn.initializers.zeros_init()
        zero_bias = nn.initializers.zeros_init()

        attn_out = nn.Dense(
            4 * self.dim,
            kernel_init=zero_kernel,
            bias_init=zero_bias,
            name="attn_out",
        )(h)

        ch_out = nn.Dense(
            4 * self.dim,
            kernel_init=zero_kernel,
            bias_init=zero_bias,
            name="ch_out",
        )(h)

        def split4(y):
            shift, scale, gate, log_gamma = jnp.split(y, 4, axis=-1)
            return {
                "shift": shift,
                "scale": scale,
                "gate": gate,
                "log_gamma": log_gamma,
            }

        return {
            "attn": split4(attn_out),
            "ch": split4(ch_out),
        }


# ============================================================
# MHDM
# ============================================================

class MHDM(nn.Module):
    """
    Token diffusion-style attention with tied Q=K projection.
    Input:  x (B, N, D)
    Output: y (B, N, D)

    If context is given:
      x       = query cloud  (B, M, D)
      context = support cloud (B, N, D)
      output  = (B, M, D)

    Symmetric distance logits:
      logits_ij = -||qk_i - qk_j||^2

    Optional Doob tilt:
      softmax_j(logits_ij + phi_j)

    Important:
      adding phi_i (row/source potential) would cancel under softmax.
      the nontrivial Doob tilt is on the destination/support index j.
    """
    dim: int
    num_heads: int
    bias: bool = False
    use_doob: bool = True

    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        context: Optional[Array] = None,
        log_gamma_q: Optional[Array] = None,
        log_gamma_k: Optional[Array] = None,
        phi: Optional[Array] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")

        support = x if context is None else context

        qk_proj = nn.Dense(self.dim, use_bias=self.bias, name="qk_proj")
        v_proj = nn.Dense(self.dim, use_bias=self.bias, name="v_proj")
        out_proj = nn.Dense(self.dim, use_bias=self.bias, name="out_proj")

        qk_q = qk_proj(x)        # (B, M, D)
        qk_k = qk_proj(support)  # tied Q=K
        v = v_proj(support)      # values come from support set

        if log_gamma_q is not None:
            qk_q = qk_q * jnp.exp(log_gamma_q)[:, None, :]
        if log_gamma_k is not None:
            qk_k = qk_k * jnp.exp(log_gamma_k)[:, None, :]

        qh = _to_heads(qk_q, self.num_heads)  # (B, H, M, d)
        kh = _to_heads(qk_k, self.num_heads)  # (B, H, N, d)
        vh = _to_heads(v, self.num_heads)     # (B, H, N, d)

        dot = jnp.einsum("bhmd,bhnd->bhmn", qh, kh)
        q2 = jnp.sum(qh * qh, axis=-1, keepdims=True)       # (B,H,M,1)
        k2 = jnp.sum(kh * kh, axis=-1)[:, :, None, :]       # (B,H,1,N)

        logits = 2.0 * dot - q2 - k2  # = -||q-k||^2

        # Optional Doob potential on destination/support index j
        if self.use_doob:
            if phi is None:
                phi = nn.Dense(
                    1,
                    use_bias=True,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                    name="doob_proj",
                )(support)[..., 0]  # (B, N)
            logits = logits + phi[:, None, None, :]

        attn = nn.softmax(logits, axis=-1)
        w = jnp.einsum("bhmn,bhnd->bhmd", attn, vh)

        out = _from_heads(w)
        out = out_proj(out)

        aux = {
            "attn": attn,
            "logits": logits,
        }
        if phi is not None:
            aux["phi"] = phi

        return out, aux


# ============================================================
# MHDM_ (channel version, close to your PyTorch version)
# ============================================================

class MHDM_(nn.Module):
    """
    Channel version: transpose token/channel axes and run the same mechanism.
    Input:  x (B, N, D)
    Output: y (B, N, D)

    Important:
      Here the Dense layers act on the last dim N, so they are shape-dependent
      on token length at init time, exactly like your lazy PyTorch logic.
    """
    num_heads: int = 1
    bias: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Dict[str, Array]]:
        B, N, D = x.shape
        if N % self.num_heads != 0:
            raise ValueError(f"N={N} must be divisible by num_heads={self.num_heads}")

        y = x.transpose(0, 2, 1)  # (B, D, N)

        qk = nn.Dense(N, use_bias=self.bias, name="qk_proj")(y)  # (B, D, N)
        v = nn.Dense(N, use_bias=self.bias, name="v_proj")(y)    # (B, D, N)

        qh = _to_heads(qk, self.num_heads)  # treats D as sequence length, N as embedding
        vh = _to_heads(v, self.num_heads)

        dot = jnp.einsum("bhmd,bhnd->bhmn", qh, qh)
        q2 = jnp.sum(qh * qh, axis=-1, keepdims=True)
        logits = 2.0 * dot - q2 - jnp.swapaxes(q2, -1, -2)

        attn = nn.softmax(logits, axis=-1)
        w = jnp.einsum("bhmn,bhnd->bhmd", attn, vh)

        y = _from_heads(w)  # (B, D, N)
        y = nn.Dense(N, use_bias=self.bias, name="out_proj")(y)

        return y.transpose(0, 2, 1), {"attn": attn, "logits": logits}


# ============================================================
# ChannelDiffusion
# ============================================================

class ChannelDiffusion(nn.Module):
    """
    Learns a feature-space transform W (D->D) and computes diffusion logits
    as negative squared Euclidean distances BETWEEN CHANNEL VECTORS across tokens.

    Input:  x (B, N, D)
    Output: y (B, N, D)

    Params depend on D, not on N.
    """
    dim: int
    num_heads: int = 8
    bias: bool = False
    temperature: bool = True

    @nn.compact
    def __call__(
        self,
        x: Array,
        *,
        log_gamma: Optional[Array] = None,
    ) -> Tuple[Array, Dict[str, Array]]:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")

        B, N, D = x.shape
        dh = self.dim // self.num_heads

        qk = nn.Dense(self.dim, use_bias=self.bias, name="qk_proj")(x)
        v = nn.Dense(self.dim, use_bias=self.bias, name="v_proj")(x)

        if log_gamma is not None:
            qk = qk * jnp.exp(log_gamma)[:, None, :]

        # Channels-as-sequence
        qk = qk.transpose(0, 2, 1)  # (B, D, N)
        v = v.transpose(0, 2, 1)    # (B, D, N)

        # (B, H, dh, N)
        qk = qk.reshape(B, self.num_heads, dh, N)
        v = v.reshape(B, self.num_heads, dh, N)

        dot = jnp.einsum("bhcn,bhdn->bhcd", qk, qk)
        q2 = jnp.sum(qk * qk, axis=-1)  # (B,H,dh)
        logits = 2.0 * dot - q2[:, :, :, None] - q2[:, :, None, :]

        # same stabilization idea as your PyTorch version
        logits = logits / math.sqrt(max(N, 1))

        if self.temperature:
            tau = self.param(
                "tau",
                lambda key, shape: jnp.ones(shape, dtype=jnp.float32),
                (self.num_heads, 1, 1),
            )
            logits = logits * tau[None, ...]

        attn = nn.softmax(logits, axis=-1)
        w = jnp.einsum("bhcd,bhdn->bhcn", attn, v)

        out = w.reshape(B, D, N).transpose(0, 2, 1)
        out = nn.Dense(self.dim, use_bias=self.bias, name="out_proj")(out)

        return out, {"attn": attn, "logits": logits}


# ============================================================
# Diffusion_Block with time conditioning
# ============================================================

class DiffusionBlock(nn.Module):
    """
    Flax/JAX analogue of your PyTorch block, but time-conditioning ready.

    Time conditioning is injected by:
      - AdaLN shift/scale
      - DiT-style residual gates
      - log_gamma(t) metric scaling
    """
    dim: int
    num_heads: int
    channel_heads: int = 8
    dropout: float = 0.0
    bias: bool = False
    use_time: bool = True
    use_doob: bool = True

    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Optional[Array] = None,
        *,
        deterministic: bool = True,
    ) -> Tuple[Array, Dict[str, Any]]:
        drop = nn.Dropout(rate=self.dropout)

        norm1 = nn.LayerNorm(use_bias=True, use_scale=True, name="norm1")
        norm2 = nn.LayerNorm(use_bias=True, use_scale=True, name="norm2")

        x1 = norm1(x)
        x2_input = None

        # default "no conditioning"
        B = x.shape[0]
        zeros = jnp.zeros((B, self.dim), dtype=x.dtype)
        mods = {
            "attn": {"shift": zeros, "scale": zeros, "gate": zeros, "log_gamma": zeros},
            "ch": {"shift": zeros, "scale": zeros, "gate": zeros, "log_gamma": zeros},
        }

        if self.use_time:
            if t is None:
                raise ValueError("t must be provided when use_time=True")
            mods = AdaLNTimeConditioner(self.dim, name="time_cond")(t)
            x1 = modulate(x1, mods["attn"]["shift"], mods["attn"]["scale"])

        attn_out, attn_aux = MHDM(
            dim=self.dim,
            num_heads=self.num_heads,
            bias=self.bias,
            use_doob=self.use_doob,
            name="attn",
        )(
            x1,
            log_gamma_q=mods["attn"]["log_gamma"] if self.use_time else None,
            log_gamma_k=mods["attn"]["log_gamma"] if self.use_time else None,
        )

        gamma1 = self.param(
            "gamma1",
            lambda key, shape: 1e-4 * jnp.ones(shape, dtype=jnp.float32),
            (self.dim,),
        )

        x = x + drop(attn_out, deterministic=deterministic) * mods["attn"]["gate"][:, None, :] * gamma1

        x2 = norm2(x)
        if self.use_time:
            x2 = modulate(x2, mods["ch"]["shift"], mods["ch"]["scale"])

        ch_out, ch_aux = ChannelDiffusion(
            dim=self.dim,
            num_heads=self.channel_heads,
            bias=self.bias,
            temperature=True,
            name="ch_attn",
        )(
            x2,
            log_gamma=mods["ch"]["log_gamma"] if self.use_time else None,
        )

        gamma2 = self.param(
            "gamma2",
            lambda key, shape: 1e-4 * jnp.ones(shape, dtype=jnp.float32),
            (self.dim,),
        )

        x = x + drop(ch_out, deterministic=deterministic) * mods["ch"]["gate"][:, None, :] * gamma2

        aux = {
            "mods": mods,
            "attn": attn_aux,
            "channel": ch_aux,
        }
        return x, aux


# ============================================================
# Minimal Optax training scaffold
# ============================================================

class TrainState(train_state.TrainState):
    pass


def create_train_state(
    rng: Array,
    model: nn.Module,
    x_shape: Tuple[int, int, int],
    learning_rate: float = 1e-3,
) -> TrainState:
    B, N, D = x_shape
    x = jnp.zeros((B, N, D), dtype=jnp.float32)
    t = jnp.zeros((B,), dtype=jnp.float32)

    variables = model.init(rng, x, t, deterministic=True)
    tx = optax.adamw(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


@jax.jit
def train_step(
    state: TrainState,
    x: Array,
    t: Array,
    target: Array,
    rng: Array,
):
    def loss_fn(params):
        (y, aux) = state.apply_fn(
            {"params": params},
            x,
            t,
            deterministic=False,
            rngs={"dropout": rng},
        )
        loss = jnp.mean((y - target) ** 2)
        return loss, {"loss": loss, "aux": aux}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics
