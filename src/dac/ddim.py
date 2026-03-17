import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from mhdm import DiffusionBlock


Array = jnp.ndarray


# ============================================================
# Schedule helpers
# ============================================================

def make_beta_schedule(
    num_train_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> Array:
    return jnp.linspace(beta_start, beta_end, num_train_steps, dtype=jnp.float32)


@dataclass
class DDIMSchedule:
    betas: Array                  # (T,)
    alphas: Array                 # (T,)
    alpha_bars: Array             # (T,)
    num_train_steps: int


def make_ddim_schedule(
    num_train_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> DDIMSchedule:
    betas = make_beta_schedule(num_train_steps, beta_start, beta_end)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas, axis=0)
    return DDIMSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        num_train_steps=num_train_steps,
    )


def extract(a: Array, t: Array, x_shape: Tuple[int, ...]) -> Array:
    """
    a: (T,)
    t: (B,) int32
    returns shape (B, 1, 1, ..., 1) broadcastable to x_shape
    """
    B = t.shape[0]
    out = a[t]
    return out.reshape((B,) + (1,) * (len(x_shape) - 1))


def make_ddim_timesteps(
    num_train_steps: int,
    num_sampling_steps: int,
) -> Array:
    """
    Uniform stride schedule for DDIM sampling.
    Returns integer timesteps in ascending order.
    """
    ts = jnp.linspace(0, num_train_steps - 1, num_sampling_steps)
    ts = jnp.round(ts).astype(jnp.int32)
    ts = jnp.unique(ts)
    return ts


# ============================================================
# Forward diffusion
# ============================================================

def q_sample(
    schedule: DDIMSchedule,
    x0: Array,
    t: Array,
    noise: Array,
) -> Array:
    """
    x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps
    """
    a_bar_t = extract(schedule.alpha_bars, t, x0.shape)
    return jnp.sqrt(a_bar_t) * x0 + jnp.sqrt(1.0 - a_bar_t) * noise


def predict_x0_from_eps(
    schedule: DDIMSchedule,
    x_t: Array,
    t: Array,
    eps: Array,
) -> Array:
    a_bar_t = extract(schedule.alpha_bars, t, x_t.shape)
    return (x_t - jnp.sqrt(1.0 - a_bar_t) * eps) / jnp.sqrt(a_bar_t)


# ============================================================
# Model
# ============================================================

class DDIMModel(nn.Module):
    """
    Wraps DiffusionBlock(s) into a DDIM/DDPM-style epsilon-predictor.

    Input:
      x_t : (B, N, input_dim)
      t   : (B,) integer timesteps in [0, T-1]

    Output:
      eps_theta(x_t, t): (B, N, input_dim)

    Notes:
      - This uses DiffusionBlock as the denoising backbone.
      - DDIM sampling is implemented below in ddim_step / sample_ddim.
      - For now this is epsilon-prediction, which is the standard simple choice.
    """
    input_dim: int
    model_dim: int
    depth: int
    num_heads: int
    channel_heads: int = 8
    dropout: float = 0.0
    num_train_steps: int = 1000
    use_doob: bool = True
    bias: bool = False

    @nn.compact
    def __call__(
        self,
        x_t: Array,
        t: Array,
        *,
        deterministic: bool = True,
    ) -> Tuple[Array, Dict[str, Any]]:
        # Normalize timestep to [0,1] for the time conditioner inside DiffusionBlock
        t = jnp.asarray(t, dtype=jnp.float32)
        t_cont = t / jnp.maximum(self.num_train_steps - 1, 1)

        # Lift input coordinates/features into model space
        h = nn.Dense(self.model_dim, use_bias=True, name="in_proj")(x_t)

        aux_blocks = []
        for i in range(self.depth):
            h, aux = DiffusionBlock(
                dim=self.model_dim,
                num_heads=self.num_heads,
                channel_heads=self.channel_heads,
                dropout=self.dropout,
                bias=self.bias,
                use_time=True,
                use_doob=self.use_doob,
                name=f"block_{i}",
            )(h, t_cont, deterministic=deterministic)
            aux_blocks.append(aux)

        h = nn.LayerNorm(use_bias=True, use_scale=True, name="final_norm")(h)

        # Predict epsilon in input space
        eps = nn.Dense(
            self.input_dim,
            use_bias=True,
            name="out_proj",
        )(h)

        aux = {
            "blocks": aux_blocks,
        }
        return eps, aux


# ============================================================
# Optax train state
# ============================================================

class TrainState(train_state.TrainState):
    pass


def create_train_state(
    rng: Array,
    model: DDIMModel,
    example_x: Array,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
) -> TrainState:
    B = example_x.shape[0]
    example_t = jnp.zeros((B,), dtype=jnp.int32)
    variables = model.init(rng, example_x, example_t, deterministic=True)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


# ============================================================
# DDPM/DDIM training loss
# ============================================================

def diffusion_loss(
    params: Dict[str, Any],
    apply_fn,
    schedule: DDIMSchedule,
    x0: Array,
    rng: Array,
    *,
    deterministic: bool = True,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Standard epsilon-prediction loss:
      E_{t,eps}[ || eps - eps_theta(x_t, t) ||^2 ]
    """
    B = x0.shape[0]
    rng_t, rng_noise, rng_drop = jax.random.split(rng, 3)

    t = jax.random.randint(
        rng_t,
        shape=(B,),
        minval=0,
        maxval=schedule.num_train_steps,
        dtype=jnp.int32,
    )

    noise = jax.random.normal(rng_noise, x0.shape)
    x_t = q_sample(schedule, x0, t, noise)

    (eps_pred, aux) = apply_fn(
        {"params": params},
        x_t,
        t,
        deterministic=deterministic,
        rngs={"dropout": rng_drop},
    )

    loss = jnp.mean((eps_pred - noise) ** 2)
    metrics = {
        "loss": loss,
        "eps_mse": loss,
    }
    return loss, metrics


@jax.jit
def train_step(
    state: TrainState,
    schedule: DDIMSchedule,
    x0: Array,
    rng: Array,
) -> Tuple[TrainState, Dict[str, Array]]:
    def loss_fn(params):
        return diffusion_loss(
            params,
            state.apply_fn,
            schedule,
            x0,
            rng,
            deterministic=False,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


# ============================================================
# DDIM sampler
# ============================================================

def ddim_step(
    params: Dict[str, Any],
    apply_fn,
    schedule: DDIMSchedule,
    x_t: Array,
    t: Array,
    t_prev: Array,
    rng: Optional[Array],
    *,
    eta: float = 0.0,
    clip_x0: Optional[float] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    One DDIM step.

    Inputs:
      x_t    : (B, N, D)
      t      : (B,) current integer timestep
      t_prev : (B,) previous integer timestep (usually smaller)
      eta    : 0.0 => deterministic DDIM
               >0  => stochastic variant approaching DDPM-like behavior

    Returns:
      x_prev
    """
    eps_pred, aux = apply_fn(
        {"params": params},
        x_t,
        t,
        deterministic=True,
    )

    a_bar_t = extract(schedule.alpha_bars, t, x_t.shape)
    a_bar_prev = extract(schedule.alpha_bars, t_prev, x_t.shape)

    x0_pred = predict_x0_from_eps(schedule, x_t, t, eps_pred)
    if clip_x0 is not None:
        x0_pred = jnp.clip(x0_pred, -clip_x0, clip_x0)

    # DDIM sigma_t
    sigma_t = eta * jnp.sqrt((1.0 - a_bar_prev) / (1.0 - a_bar_t)) * jnp.sqrt(
        1.0 - a_bar_t / a_bar_prev
    )

    # direction pointing to x_t
    c_t = jnp.sqrt(jnp.maximum(1.0 - a_bar_prev - sigma_t**2, 0.0))

    if rng is None:
        z = jnp.zeros_like(x_t)
    else:
        z = jax.random.normal(rng, x_t.shape)

    x_prev = (
        jnp.sqrt(a_bar_prev) * x0_pred
        + c_t * eps_pred
        + sigma_t * z
    )

    stats = {
        "x0_pred": x0_pred,
        "eps_pred": eps_pred,
        "sigma_t": sigma_t,
    }
    stats.update(aux)
    return x_prev, stats


def sample_ddim(
    params: Dict[str, Any],
    apply_fn,
    schedule: DDIMSchedule,
    rng: Array,
    sample_shape: Tuple[int, int, int],
    *,
    num_sampling_steps: int = 50,
    eta: float = 0.0,
    clip_x0: Optional[float] = None,
) -> Tuple[Array, Dict[str, Any]]:
    """
    Full DDIM sampler.

    sample_shape: (B, N, D)
    """
    B = sample_shape[0]
    x = jax.random.normal(rng, sample_shape)

    ddim_ts = make_ddim_timesteps(schedule.num_train_steps, num_sampling_steps)

    # iterate from large t to small t
    all_stats = []
    step_rngs = jax.random.split(rng, len(ddim_ts))

    for i in range(len(ddim_ts) - 1, -1, -1):
        t_scalar = ddim_ts[i]
        t_prev_scalar = 0 if i == 0 else ddim_ts[i - 1]

        t = jnp.full((B,), t_scalar, dtype=jnp.int32)
        t_prev = jnp.full((B,), t_prev_scalar, dtype=jnp.int32)

        # no noise added at the final step when eta=0, but leaving rng harmless
        x, stats = ddim_step(
            params=params,
            apply_fn=apply_fn,
            schedule=schedule,
            x_t=x,
            t=t,
            t_prev=t_prev,
            rng=step_rngs[i] if eta > 0.0 else None,
            eta=eta,
            clip_x0=clip_x0,
        )
        all_stats.append(stats)

    return x, {
        "timesteps": ddim_ts,
        "steps": all_stats,
    }
