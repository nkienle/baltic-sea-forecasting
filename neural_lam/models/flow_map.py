"""
Flow map forecasting model (FlowMap), on the GraphDiff backbone.

Ported from score-based-ensemble-filter/forecasting/models/flow_map.py, which
follows Boffi et al., "How to build a consistency model: learning flow maps
via self-distillation" (github.com/nmboffi/flow-maps). Time 0 is noise, time 1
is data and the interpolant is I_t = (1 - t) x0 + t x1, as in FM. The data x1
is the (normalized residual of the) next state, conditioned on the previous
state(s) and forcing exactly as FM.

The flow map is X_{s,t}(x) = x + (t - s) phi_{s,t}(x); its diagonal
phi_{t,t} = b_t is the FM velocity. The backbone takes s through its usual
time embedding and dt = t - s through a zero-initialised `map_dt`, so an FM
checkpoint loads as a flow map with phi_{s,t} = b_s (one Euler step).

Losses (per sample):
- diagonal (s = t): |phi_{t,t}(I_t) - (x1 - x0)|^2, or the teacher's b_t(I_t).
- lsd: |d/dtau X_{s,s+tau d}(I_s) - d * b_t(sg X_{s,t}(I_s))|^2, d = t - s.
- esd: |d/dtau X_{s+tau d,t}(I_s) + grad_x X_{s,t}(I_s) [d * b_s(I_s)]|^2.
- psd: |phi_{s,t} - sg[(1-h) phi_{s,u} + h phi_{u,t}(X_{s,u})]|^2,
  u = h s + (1-h) t.
Diagonal and off-diagonal terms are weighted by a bounded learned
log-variance w(s, t) as exp(-w) L + w (|w| <= sqrt(2)).

--fmap_objective selects what is trained:
- map: the flow-map loss above, on the true previous states.
- crps: almost fair CRPS of --n_ens_train ensemble members, each step of the
  autoregressive rollout a single jump X_{0,1}(z), z ~ N(0, I).
- map+crps: crps + --fmap_crps_map_weight * map (map on the first step), which
  keeps phi a consistent flow map (multi-step sampling stays valid).

With --fmap_ema an exponential moving average of the backbone weights is kept and
used in validation and test (swapped in at the start, out at the end).
"""

# Standard library
import contextlib
import math
from typing import Union

# Third-party
import torch
from torch import nn

# First-party
from neural_lam import metrics
from neural_lam.models.FM import FM

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore

FMAP_LOSSES = ("lsd", "esd", "psd")
FMAP_OBJECTIVES = ("map", "crps", "map+crps")
STOPGRAD_OPTIONS = {
    "lsd": ("convex", "none"),
    "esd": ("full", "convex", "none"),
    "psd": ("convex", "none"),
}

#: Defaults, used by the parser and as fallbacks when a model is rebuilt from
#: a namespace without these flags (e.g. an older checkpoint).
FLOW_MAP_DEFAULTS = {
    "fmap_loss": "lsd",
    "fmap_psd_type": "uniform",
    "fmap_stopgrad": "auto",
    "fmap_diag_fraction": 0.75,
    "fmap_pair_mode": "auto",
    "fmap_tmin": 0.0,
    "fmap_tmax": 1.0,
    "fmap_max_gap": [1.0],
    "fmap_max_gap_ramp": [0.0, 0.5],
    "fmap_max_gap_shape": "geometric",
    "fmap_learned_weight": True,
    "fmap_w_log_every": 50,
    "fmap_teacher_ckpt": None,
    "fmap_init_from_teacher": False,
    "fmap_diag_target": "data",
    "fmap_offdiag_target": "self",
    "fmap_sample_steps": 1,
    "fmap_val_sample_steps_extra": 0,
    "fmap_dt_embedding": "positional",
    "fmap_objective": "map",
    "fmap_crps_map_weight": 0.5,
    "n_ens_train": 2,
    "fmap_ema": False,
    "fmap_ema_decay": 0.9999,
}


def _list_of_floats(arg):
    return [float(v) for v in arg.split(",")]


def _str2bool(arg):
    if arg.lower() in ("true", "1", "yes"):
        return True
    if arg.lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"expected true/false, got {arg!r}")


def add_flow_map_args(parser):
    """Add the --fmap_* flags of the FlowMap model"""
    d = FLOW_MAP_DEFAULTS
    g = parser.add_argument_group("flow map (FlowMap)")
    g.add_argument(
        "--fmap_objective",
        choices=FMAP_OBJECTIVES,
        default=d["fmap_objective"],
        help="map: flow-map loss; crps: almost fair CRPS of 1-jump ensemble "
        "rollouts; map+crps: both (default: map). The flow_map_* schedulers "
        "set this per part.",
    )
    g.add_argument(
        "--fmap_crps_map_weight",
        type=float,
        default=d["fmap_crps_map_weight"],
        help="Weight of the flow-map loss in map+crps (default: 0.5)",
    )
    g.add_argument(
        "--n_ens_train",
        type=int,
        default=d["n_ens_train"],
        help="Ensemble members per sample for the CRPS objective (default: 2)",
    )
    g.add_argument(
        "--fmap_loss",
        choices=FMAP_LOSSES,
        default=d["fmap_loss"],
        help="Off-diagonal self-distillation loss: lsd (Lagrangian), esd "
        "(Eulerian) or psd (progressive) (default: lsd)",
    )
    g.add_argument(
        "--fmap_psd_type",
        choices=("uniform", "midpoint"),
        default=d["fmap_psd_type"],
        help="PSD intermediate time: u = h s + (1-h) t with h ~ U(0,1), or "
        "the midpoint",
    )
    g.add_argument(
        "--fmap_stopgrad",
        default=d["fmap_stopgrad"],
        help="Stop-gradient scheme. auto = lsd:convex, esd:full, psd:convex. "
        "Also lsd/psd: none; esd: convex, none.",
    )
    g.add_argument(
        "--fmap_diag_fraction",
        type=float,
        default=d["fmap_diag_fraction"],
        help="Weight of the diagonal (flow matching) term (default: 0.75)",
    )
    g.add_argument(
        "--fmap_pair_mode",
        choices=("auto", "split", "both", "alternate"),
        default=d["fmap_pair_mode"],
        help="split: a fraction diag_fraction of the batch is diagonal, the "
        "rest off-diagonal (needs batch_size >= 2). both: every sample gets "
        "a diagonal and an off-diagonal term, weighted diag_fraction and "
        "1 - diag_fraction. alternate: each step is all diagonal (with "
        "probability diag_fraction) or all off-diagonal, the same expected "
        "gradient as split with about half the peak memory of both. auto: "
        "split for batch_size >= 4, else both.",
    )
    g.add_argument("--fmap_tmin", type=float, default=d["fmap_tmin"])
    g.add_argument("--fmap_tmax", type=float, default=d["fmap_tmax"])
    g.add_argument(
        "--fmap_max_gap",
        type=_list_of_floats,
        default=d["fmap_max_gap"],
        help="Largest |t - s| of the off-diagonal pairs: one value, or "
        "start,end to ramp it over the run (e.g. 0.05,1). Validation uses end",
    )
    g.add_argument(
        "--fmap_max_gap_ramp",
        type=_list_of_floats,
        default=d["fmap_max_gap_ramp"],
        help="Fractions a,b of the run over which --fmap_max_gap goes from "
        "start to end",
    )
    g.add_argument(
        "--fmap_max_gap_shape",
        choices=("linear", "geometric"),
        default=d["fmap_max_gap_shape"],
    )
    g.add_argument(
        "--fmap_learned_weight",
        type=_str2bool,
        default=d["fmap_learned_weight"],
        help="Bounded learned log-variance weight w(s, t) (true/false)",
    )
    g.add_argument(
        "--fmap_w_log_every",
        type=int,
        default=d["fmap_w_log_every"],
        help="Log w(s, t) at fixed probes every this many optimizer steps",
    )
    g.add_argument(
        "--fmap_teacher_ckpt",
        type=str,
        default=d["fmap_teacher_ckpt"],
        help="Lightning checkpoint of a trained FM model to distil from",
    )
    g.add_argument(
        "--fmap_init_from_teacher",
        action="store_true",
        help="Initialise the backbone with the teacher's weights "
        "(phi_{s,t} = b_s at the start)",
    )
    g.add_argument(
        "--fmap_diag_target",
        choices=("data", "teacher"),
        default=d["fmap_diag_target"],
        help="Diagonal target: x1 - x0 (data) or the teacher's b_t",
    )
    g.add_argument(
        "--fmap_offdiag_target",
        choices=("self", "teacher"),
        default=d["fmap_offdiag_target"],
        help="Velocity in the lsd/esd targets: the model's own "
        "(self-distillation) or the teacher's (LMD/EMD distillation)",
    )
    g.add_argument(
        "--fmap_sample_steps",
        type=int,
        default=d["fmap_sample_steps"],
        help="Flow-map jumps from noise to data when sampling, in validation "
        "and test (default: 1)",
    )
    g.add_argument(
        "--fmap_val_sample_steps_extra",
        type=int,
        default=d["fmap_val_sample_steps_extra"],
        help="If > 0, also log validation ensemble CRPS / spread-skill with "
        "this many jumps on the first batch, to compare with 1 jump",
    )
    g.add_argument(
        "--fmap_dt_embedding",
        choices=("positional", "fourier"),
        default=d["fmap_dt_embedding"],
        help="Embedding of dt = t - s (default: positional, smooth in t)",
    )
    g.add_argument(
        "--fmap_ema",
        action="store_true",
        help="Keep an EMA of the backbone weights for validation/test",
    )
    g.add_argument(
        "--fmap_ema_decay",
        type=float,
        default=d["fmap_ema_decay"],
        help="EMA decay; the effective decay is min(ema_decay, "
        "(1 + n) / (10 + n)) n steps after the EMA was (re)started",
    )
    return parser


def _bc(t, x):
    """Broadcast times (B,) against x (B, N, d)"""
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def _take(x, idx):
    """Index the batch dimension of a tensor or a tuple of tensors/None"""
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(_take(xi, idx) for xi in x)
    return x[idx]


# ----------------------------------------------------------------------------
# Learned loss weight
class MPPositionalEmbedding(nn.Module):
    """
    Magnitude-preserving positional embedding (unit rms per channel), as in
    the reference code: log-spaced frequencies up to 1, cos and sin scaled by
    sqrt(2). (B,) -> (B, num_channels).
    """

    def __init__(self, num_channels, max_period=10000.0):
        super().__init__()
        half = num_channels // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        args = t.reshape(-1, 1).to(torch.float32) * self.freqs
        return torch.cat([args.cos(), args.sin()], dim=1) * math.sqrt(2.0)


def mp_sum(a, b, t=0.5):
    """Magnitude-preserving interpolation of a and b (EDM2)"""
    return (a * (1 - t) + b * t) / math.sqrt((1 - t) ** 2 + t**2)


class LogVarWeight(nn.Module):
    """
    Bounded learned log-variance w(s, t) for the loss exp(-w) L + w.

    The reference code's `calc_weight`: magnitude-preserving positional
    embeddings of s and t, combined with mp_sum, then a bias-free linear layer
    whose weight is kept at unit length (forced weight normalisation, see
    project). |w| <= sqrt(2), so exp(-w) stays within about [0.24, 4.1]. An
    unbounded w lets the loss minimise log L of the off-diagonal term and drop
    the diagonal one, which collapses training.
    """

    def __init__(self, num_channels=128):
        super().__init__()
        self.embed_s = MPPositionalEmbedding(num_channels)
        self.embed_t = MPPositionalEmbedding(num_channels)
        self.weight = nn.Parameter(torch.randn(num_channels))

    def forward(self, s, t):
        emb = mp_sum(self.embed_s(s), self.embed_t(t))
        w_hat = self.weight / self.weight.norm().clamp_min(1e-4)
        return emb.to(w_hat.dtype) @ w_hat / math.sqrt(w_hat.numel())

    @torch.no_grad()
    def project(self):
        """Put the stored weight back on the unit sphere. Call after each
        optimizer step (not in forward, w is evaluated several times per
        step and autograd keeps the weight for backward)."""
        self.weight.div_(self.weight.norm().clamp_min(1e-4))


# ----------------------------------------------------------------------------
class FlowMapMixin:
    """
    Flow-map losses, sampler, learned weight and EMA. Mix in before the
    Lightning model. The host model provides:
        model: the backbone (attribute name fmap_backbone_attr).
        _phi(s, t, x, cond) -> phi, shaped like x; s, t are (B,).
        _teacher_velocity(t, x, cond) -> b_t(x) from the frozen teacher.
        _sqmean(r) -> (B,), weighted mean of r^2 per sample.
    and calls setup_flow_map() at the end of __init__.
    """

    fmap_backbone_attr = "model"

    # ------------------------------------------------------------------ setup
    def fa(self, name):
        """A flow-map arg, falling back to FLOW_MAP_DEFAULTS"""
        value = getattr(self.args, name, None)
        return FLOW_MAP_DEFAULTS[name] if value is None else value

    def setup_flow_map(self):
        loss = self.fa("fmap_loss")
        assert loss in FMAP_LOSSES, f"--fmap_loss {loss} not in {FMAP_LOSSES}"
        stopgrad = self.fa("fmap_stopgrad")
        self.stopgrad = (
            STOPGRAD_OPTIONS[loss][0] if stopgrad == "auto" else stopgrad
        )
        assert self.stopgrad in STOPGRAD_OPTIONS[loss], (
            f"--fmap_stopgrad {self.stopgrad} is not valid for {loss}; "
            f"choose from {STOPGRAD_OPTIONS[loss]}"
        )
        if loss == "psd" and self.fa("fmap_offdiag_target") == "teacher":
            raise ValueError(
                "--fmap_offdiag_target teacher applies to lsd/esd; psd is a "
                "pure self-consistency loss"
            )
        gaps = self.fa("fmap_max_gap")
        assert (
            1 <= len(gaps) <= 2 and min(gaps) > 0
        ), f"--fmap_max_gap takes one or two positive values, got {gaps}"

        # Objective, switched per part by the flow_map_* schedulers
        self.fmap_objective = self.fa("fmap_objective")
        assert self.fmap_objective in FMAP_OBJECTIVES

        self.fmap_logvar = (
            LogVarWeight() if self.fa("fmap_learned_weight") else None
        )

        # EMA of the backbone parameters, kept outside the module tree (not
        # broadcast by DDP every step), saved in the checkpoint under
        # "fmap_ema". _ema_backup holds the raw weights while EMA is in use.
        self._ema_shadow = None
        self._ema_backup = None
        self._ema_n = 0
        self._ema_step = -1
        # Set by the flow_map_* schedulers when a new part starts: the EMA is
        # then restarted from the current weights at the start of training.
        self.fmap_reset_ema_on_train_start = False
        # First epoch of the current training part, for the max_gap ramp
        self.fmap_stage_start_epoch = 0

    # -------------------------------------------------------------- weights
    def fmap_backbone(self):
        return getattr(self, self.fmap_backbone_attr)

    def _ema_params(self):
        return [
            p for p in self.fmap_backbone().parameters() if p.requires_grad
        ]

    @torch.no_grad()
    def reset_ema(self):
        """(Re)start the EMA from the current backbone weights"""
        if not self.fa("fmap_ema"):
            return
        self._ema_shadow = [p.detach().clone() for p in self._ema_params()]
        self._ema_n = 0

    def _ema_to_device(self):
        """The shadow can be restored on another device than the model"""
        params = self._ema_params()
        if self._ema_shadow and self._ema_shadow[0].device != params[0].device:
            self._ema_shadow = [
                e.to(p.device, p.dtype)
                for e, p in zip(self._ema_shadow, params)
            ]

    @torch.no_grad()
    def update_ema(self):
        if self._ema_shadow is None:
            self.reset_ema()
            return
        self._ema_to_device()
        n = self._ema_n
        decay = min(self.fa("fmap_ema_decay"), (1 + n) / (10 + n))
        torch._foreach_lerp_(self._ema_shadow, self._ema_params(), 1 - decay)
        self._ema_n += 1

    @torch.no_grad()
    def swap_in_ema(self):
        """Use the EMA weights (validation, test); undo with swap_out_ema"""
        if self._ema_shadow is None or self._ema_backup is not None:
            return
        self._ema_to_device()
        params = self._ema_params()
        self._ema_backup = [p.detach().clone() for p in params]
        for p, e in zip(params, self._ema_shadow):
            p.copy_(e)

    @torch.no_grad()
    def swap_out_ema(self):
        if self._ema_backup is None:
            return
        for p, b in zip(self._ema_params(), self._ema_backup):
            p.copy_(b)
        self._ema_backup = None

    @contextlib.contextmanager
    def dropout_off(self):
        """Backbone dropout off for the off-diagonal terms (as the reference
        code); the diagonal (flow matching) term keeps it. Only the backbone
        is switched, self.training is unchanged."""
        net = self.fmap_backbone()
        was_training = net.training
        net.train(False)
        try:
            yield
        finally:
            net.train(was_training)

    def on_train_start(self):
        super().on_train_start()
        if self.fa("fmap_ema") and (
            self._ema_shadow is None or self.fmap_reset_ema_on_train_start
        ):
            self.reset_ema()
            self.fmap_reset_ema_on_train_start = False

    def on_train_batch_start(self, batch, batch_idx):
        self._fmap_batch_idx = batch_idx
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        # Once per optimizer step (several batches with grad accumulation)
        if self.global_step != self._ema_step:
            self._ema_step = self.global_step
            if self.fa("fmap_ema"):
                self.update_ema()
            if self.fmap_logvar is not None:
                self.fmap_logvar.project()
            self._log_w_probes()

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if torch.cuda.is_available() and self.global_rank == 0:
            print(
                "[flow map] peak GPU memory "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                f"(objective {self.fmap_objective})",
                flush=True,
            )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.swap_in_ema()

    def on_validation_epoch_end(self):
        # Before the checkpoint callbacks run (on_validation_end), so that the
        # raw weights are what is saved as the model weights
        self.swap_out_ema()
        super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.swap_in_ema()

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self.swap_out_ema()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        if self._ema_shadow is not None:
            checkpoint["fmap_ema"] = [e.cpu() for e in self._ema_shadow]
            checkpoint["fmap_ema_n"] = self._ema_n
        checkpoint["fmap_stage_start_epoch"] = self.fmap_stage_start_epoch

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        ema = checkpoint.get("fmap_ema")
        if self.fa("fmap_ema") and ema is not None:
            params = self._ema_params()
            if len(ema) == len(params) and all(
                e.shape == p.shape for e, p in zip(ema, params)
            ):
                self._ema_shadow = [
                    e.to(p.device, p.dtype) for e, p in zip(ema, params)
                ]
                self._ema_n = checkpoint.get("fmap_ema_n", 0)
            else:
                print(
                    "FlowMap: EMA in the checkpoint does not match the "
                    "backbone; the EMA restarts from the weights"
                )

    # ------------------------------------------------------------- flow map
    def flow_map(self, s, t, x, cond, return_phi=False):
        """X_{s,t}(x) = x + (t - s) phi_{s,t}(x); s, t are (B,)"""
        phi = self._phi(s, t, x, cond)
        X = x + _bc(t - s, x) * phi
        return (X, phi) if return_phi else X

    def velocity(self, t, x, cond):
        """b_t(x) = phi_{t,t}(x)"""
        return self._phi(t, t, x, cond)

    def flow_map_sample(self, x, n_steps, cond, tmin=0.0, tmax=1.0):
        """Jump x from tmin to tmax in n_steps uniform flow-map steps.
        Differentiable (used by the CRPS objective)."""
        B = x.shape[0]
        grid = torch.linspace(tmin, tmax, n_steps + 1, device=x.device)
        for a, b in zip(grid[:-1], grid[1:]):
            x = self.flow_map(a.expand(B), b.expand(B), x, cond)
        return x

    # --------------------------------------------------------- time sampling
    def fmap_progress(self):
        """Fraction of the current training part done, in [0, 1]"""
        trainer = getattr(self, "_trainer", None)
        if trainer is None or not trainer.max_epochs:
            return 1.0
        start = self.fmap_stage_start_epoch
        span = max(trainer.max_epochs - start, 1)
        nb = trainer.num_training_batches
        frac = 0.0
        if isinstance(nb, int) and nb > 0:
            frac = min(getattr(self, "_fmap_batch_idx", 0) / nb, 1.0)
        return min(max((trainer.current_epoch - start + frac) / span, 0), 1)

    def fmap_max_gap(self):
        """Current bound on |t - s|: the --fmap_max_gap ramp, end in eval"""
        gaps = self.fa("fmap_max_gap")
        start, end = gaps[0], gaps[-1]
        if start == end or not self.training:
            return end
        a, b = self.fa("fmap_max_gap_ramp")
        p = self.fmap_progress()
        frac = 0.0 if p <= a else 1.0 if p >= b else (p - a) / (b - a)
        if self.fa("fmap_max_gap_shape") == "geometric":
            return start * (end / start) ** frac
        return start + (end - start) * frac

    def _generator(self, device):
        """Per-rank generator of the host model, if it has one"""
        get = getattr(self, "_get_generator", None)
        return get(device) if get is not None else None

    def _rand(self, n, device):
        return torch.rand(n, device=device, generator=self._generator(device))

    def _pair(self, n, device):
        """n pairs s < t, uniform on {t - s <= max_gap}"""
        lo, hi = self.fa("fmap_tmin"), self.fa("fmap_tmax")
        span = hi - lo
        gap = self.fmap_max_gap()
        if gap >= span:
            a = lo + span * self._rand(n, device)
            b = lo + span * self._rand(n, device)
            return torch.minimum(a, b), torch.maximum(a, b)
        # d = t - s has density prop. to (span - d) on [0, gap] (inverse CDF)
        u = self._rand(n, device)
        d = (
            span
            - (span**2 - u * (2 * span * gap - gap**2)).clamp_min(0).sqrt()
        )
        s = lo + (span - d) * self._rand(n, device)
        return s, s + d

    def _diag_times(self, n, device):
        lo, hi = self.fa("fmap_tmin"), self.fa("fmap_tmax")
        return lo + (hi - lo) * self._rand(n, device)

    def _pair_mode(self, B):
        mode = self.fa("fmap_pair_mode")
        if mode == "auto":
            return "split" if B >= 4 else "both"
        if mode == "split":
            assert B >= 2, "--fmap_pair_mode split needs batch_size >= 2"
        return mode

    # --------------------------------------------------------------- losses
    @staticmethod
    def _interp(t, x0, x1):
        return (1 - _bc(t, x0)) * x0 + _bc(t, x0) * x1

    def _weighted(self, L, s, t, logs=None, name=None):
        if self.fmap_logvar is None:
            return L
        w = self.fmap_logvar(s, t).to(L.dtype)
        if logs is not None:
            wd = w.detach()
            logs[f"fmap_w_{name}_mean"] = wd.mean()
            logs[f"fmap_w_{name}_min"] = wd.min()
            logs[f"fmap_w_{name}_max"] = wd.max()
        return torch.exp(-w) * L + w

    #: (s, t) probes of w(s, t) logged as fmap_w/s{s}_t{t}
    W_PROBES = (
        (0.1, 0.1),
        (0.5, 0.5),
        (0.9, 0.9),
        (0.5, 0.55),
        (0.25, 0.75),
        (0.0, 1.0),
    )

    @torch.no_grad()
    def _log_w_probes(self):
        every = self.fa("fmap_w_log_every")
        if (
            not every
            or self.global_step % every
            or getattr(self, "_trainer", None) is None
        ):
            return
        if torch.cuda.is_available():
            # Peak GPU memory so far, for sizing batch/model
            self.log(
                "gpu_max_mem_gb",
                torch.cuda.max_memory_allocated() / 2**30,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
        if self.fmap_logvar is None:
            return
        device = self.fmap_logvar.weight.device
        s = torch.tensor([p[0] for p in self.W_PROBES], device=device)
        t = torch.tensor([p[1] for p in self.W_PROBES], device=device)
        w = self.fmap_logvar(s, t)
        for (ps, pt), value in zip(self.W_PROBES, w):
            self.log(
                f"fmap_w/s{ps}_t{pt}",
                value,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )

    def diag_loss(self, t, x0, x1, cond):
        It = self._interp(t, x0, x1)
        phi = self.velocity(t, It, cond)
        if self.fa("fmap_diag_target") == "teacher":
            target = self.teacher_velocity(t, It, cond)
        else:
            target = x1 - x0
        return self._sqmean(phi - target)

    def _offdiag_velocity(self, t, x, cond, grad):
        if self.fa("fmap_offdiag_target") == "teacher":
            return self.teacher_velocity(t, x.detach(), cond)
        if grad:
            return self.velocity(t, x, cond)
        with torch.no_grad():
            return self.velocity(t, x.detach(), cond)

    def lsd_loss(self, s, t, x0, x1, cond):
        Is = self._interp(s, x0, x1)
        d = t - s
        X, dX = torch.func.jvp(
            lambda tt: self.flow_map(s, tt, Is, cond), (t,), (d,)
        )
        if self.stopgrad == "convex":
            b = self._offdiag_velocity(t, X.detach(), cond, grad=False)
        else:  # none
            b = self._offdiag_velocity(t, X, cond, grad=True)
        # Tangent d instead of 1, normalised by |d|: the Boffi loss
        r = (dX - _bc(d, X) * b) / _bc(d.abs().clamp_min(1e-6), X)
        return self._sqmean(r)

    def esd_loss(self, s, t, x0, x1, cond):
        Is = self._interp(s, x0, x1)
        d = t - s
        _, dsX = torch.func.jvp(
            lambda ss: self.flow_map(ss, t, Is, cond), (s,), (d,)
        )
        b = self._offdiag_velocity(s, Is, cond, grad=(self.stopgrad == "none"))
        ctx = (
            torch.no_grad()
            if self.stopgrad == "full"
            else contextlib.nullcontext()
        )
        with ctx:
            _, gX = torch.func.jvp(
                lambda xx: self.flow_map(s, t, xx, cond),
                (Is,),
                (_bc(d, Is) * b,),
            )
        r = (dsX + gX) / _bc(d.abs().clamp_min(1e-6), Is)
        return self._sqmean(r)

    def psd_loss(self, s, t, x0, x1, cond):
        Is = self._interp(s, x0, x1)
        B = s.shape[0]
        if self.fa("fmap_psd_type") == "midpoint":
            h = torch.full((B,), 0.5, device=s.device, dtype=s.dtype)
        else:
            h = self._rand(B, s.device).to(s.dtype)
        u = h * s + (1 - h) * t
        phi_st = self._phi(s, t, Is, cond)
        ctx = (
            torch.no_grad()
            if self.stopgrad == "convex"
            else contextlib.nullcontext()
        )
        with ctx:
            X_su, phi_su = self.flow_map(s, u, Is, cond, return_phi=True)
            phi_ut = self._phi(u, t, X_su, cond)
            target = (1 - _bc(h, Is)) * phi_su + _bc(h, Is) * phi_ut
        return self._sqmean(phi_st - target)

    def flow_map_loss(self, x1, cond):
        """Flow-map loss for data x1 (B, N, d), per sample (B,)"""
        B, device = x1.shape[0], x1.device
        x0 = torch.randn(
            x1.shape,
            device=device,
            dtype=x1.dtype,
            generator=self._generator(device),
        )
        frac = self.fa("fmap_diag_fraction")
        off_fn = getattr(self, f"{self.fa('fmap_loss')}_loss")
        logs = {}

        mode = self._pair_mode(B)
        if mode == "alternate":
            # One term per step. Ranks may draw differently, which DDP
            # allows: every parameter is used by both terms.
            if self._rand(1, device).item() < frac:
                t_d = self._diag_times(B, device)
                L_d = self.diag_loss(t_d, x0, x1, cond)
                logs["fmap_diag"] = L_d.mean()
                loss = self._weighted(L_d, t_d, t_d, logs, "diag")
            else:
                s, t = self._pair(B, device)
                with self.dropout_off():
                    L_o = off_fn(s, t, x0, x1, cond)
                logs[f"fmap_{self.fa('fmap_loss')}"] = L_o.mean()
                loss = self._weighted(L_o, s, t, logs, "off")
        elif mode == "both":
            t_d = self._diag_times(B, device)
            L_d = self.diag_loss(t_d, x0, x1, cond)
            s, t = self._pair(B, device)
            with self.dropout_off():
                L_o = off_fn(s, t, x0, x1, cond)
            logs["fmap_diag"] = L_d.mean()
            logs[f"fmap_{self.fa('fmap_loss')}"] = L_o.mean()
            loss = frac * self._weighted(L_d, t_d, t_d, logs, "diag") + (
                1 - frac
            ) * self._weighted(L_o, s, t, logs, "off")
        else:
            n_diag = min(max(int(round(B * frac)), 1), B - 1)
            i, j = slice(0, n_diag), slice(n_diag, B)
            t_d = self._diag_times(n_diag, device)
            L_d = self.diag_loss(t_d, x0[i], x1[i], _take(cond, i))
            s, t = self._pair(B - n_diag, device)
            with self.dropout_off():
                L_o = off_fn(s, t, x0[j], x1[j], _take(cond, j))
            logs["fmap_diag"] = L_d.mean()
            logs[f"fmap_{self.fa('fmap_loss')}"] = L_o.mean()
            loss = torch.cat(
                [
                    self._weighted(L_d, t_d, t_d, logs, "diag"),
                    self._weighted(L_o, s, t, logs, "off"),
                ]
            )

        if self.training and len(set(self.fa("fmap_max_gap"))) > 1:
            logs["fmap_max_gap"] = torch.tensor(self.fmap_max_gap())
        self._log_flow_map(logs)
        return loss

    def _log_flow_map(self, logs):
        if getattr(self, "_trainer", None) is None:
            return
        prefix = "train" if self.training else "val"
        for key, value in logs.items():
            self.log(
                f"{prefix}_{key}",
                value.detach(),
                on_step=self.training,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )


# ----------------------------------------------------------------------------
class FlowMap(FlowMapMixin, FM):
    """
    Flow map forecasting model on the GraphDiff backbone. Conditioning,
    residual parametrization and clamping as FM (--pred_residual is
    recommended).
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
        datastore_atmosphere: Union[BaseDatastore, None],
        **kwargs,
    ):
        super().__init__(
            args,
            config,
            datastore,
            datastore_boundary,
            datastore_atmosphere,
            **kwargs,
        )
        assert self.model.map_dt is not None
        assert not self.output_std, "FlowMap does not support --output_std"
        self.setup_flow_map()

        # Keep the time embeddings out of the Muon (orthogonalized) group
        exclude = list(getattr(args, "muon_exclude_patterns", None) or [])
        for pattern in ("map_noise", "map_dt", "fmap_logvar"):
            if pattern not in exclude:
                exclude.append(pattern)
        args.muon_exclude_patterns = exclude

        # Weights over variables for the flow-map loss: the state feature
        # weights, normalised to mean 1, so that the loss is O(1)
        w = self.feature_weights / self.feature_weights.mean()
        self.register_buffer("fmap_var_weights", w, persistent=False)

        # Validation CRPS of the 1-jump ensemble, plotted like the RMSE maps
        self.val_metrics["crps_ens"] = []

        # The teacher is kept outside the module tree: frozen, not
        # checkpointed and not wrapped by DDP
        self._teacher_box = []
        path = self.fa("fmap_teacher_ckpt")
        needs_teacher = (
            self.fa("fmap_diag_target") == "teacher"
            or self.fa("fmap_offdiag_target") == "teacher"
            or self.fa("fmap_init_from_teacher")
        )
        if needs_teacher and not path:
            raise ValueError(
                "A teacher target or --fmap_init_from_teacher needs "
                "--fmap_teacher_ckpt"
            )
        if path:
            teacher = self._load_teacher(
                path,
                config,
                datastore,
                datastore_boundary,
                datastore_atmosphere,
                **kwargs,
            )
            self._teacher_box.append(teacher)
            if self.fa("fmap_init_from_teacher"):
                self._init_from_teacher(teacher)

    # ---------------------------------------------------------- teacher
    def _load_teacher(self, path, config, *datastores, **kwargs):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        targs = ckpt["hyper_parameters"]["args"]
        assert targs.model == "FM", (
            f"--fmap_teacher_ckpt must be an FM checkpoint, got model "
            f"{targs.model}"
        )
        for name in ("graph", "hidden_dim", "pred_residual", "input_steps"):
            if getattr(targs, name, None) != getattr(self.args, name, None):
                raise ValueError(
                    f"Teacher/student mismatch in --{name}: "
                    f"{getattr(targs, name, None)} vs "
                    f"{getattr(self.args, name, None)}"
                )
        teacher = FM(targs, config, *datastores, **kwargs)
        teacher.load_state_dict(ckpt["state_dict"])
        teacher.eval()
        teacher.requires_grad_(False)
        print(f"FlowMap: teacher loaded from {path}")
        return teacher

    def teacher(self, device):
        teacher = self._teacher_box[0]
        if teacher.device != device:
            teacher.to(device)
        return teacher

    @torch.no_grad()
    def teacher_velocity(self, t, x, cond):
        teacher = self.teacher(x.device)
        input_grid, boundary_forcing, atmosphere_forcing = cond
        return teacher.velocity_net(
            x, t, input_grid, boundary_forcing, atmosphere_forcing
        )

    def _init_from_teacher(self, teacher):
        result = self.model.load_state_dict(
            teacher.model.state_dict(), strict=False
        )
        missing = [
            k for k in result.missing_keys if not k.startswith("map_dt.")
        ]
        if result.unexpected_keys or missing:
            raise RuntimeError(
                f"Teacher/student mismatch: missing {missing}, unexpected "
                f"{list(result.unexpected_keys)}"
            )
        print("FlowMap: backbone initialised from the teacher")

    # ---------------------------------------------------- FlowMapMixin hooks
    def _phi(self, s, t, x, cond):
        input_grid, boundary_forcing, atmosphere_forcing = cond
        return self.model(
            x,
            s,
            input_grid,
            boundary_forcing,
            atmosphere_forcing,
            dt=t - s,
        )

    def _sqmean(self, r):
        """
        Per-sample weighted mean of r^2 (B,): mean over the (sea) grid points
        of the loss mask per variable, then mean over variables weighted by
        the normalised feature weights. Out of place (safe under jvp).
        """
        mask = self.loss_mask  # (N, d)
        per_var = (r.square() * mask).sum(dim=-2) / mask.sum(dim=-2)
        return (per_var * self.fmap_var_weights).mean(dim=-1)

    # --------------------------------------------------------- forecasting
    def _sample_next(self, prev_state, prev_prev_state, forcing,
                     boundary_forcing, atmosphere_forcing, n_steps):
        cond = (
            self.input_grid(prev_state, prev_prev_state, forcing),
            boundary_forcing,
            atmosphere_forcing,
        )
        # NOTE: The global RNG, not the per-rank generator: predict_step runs
        # under gradient checkpointing in CRPS rollouts, and checkpointing
        # only restores the global RNG state for the recomputation. Noise
        # from a separate generator would differ between the forward pass and
        # the recomputation, giving wrong gradients.
        latents = torch.randn_like(prev_state)
        y = self.flow_map_sample(latents, n_steps, cond)
        return self.target_to_state(y, prev_state)

    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        One forecast step in --fmap_sample_steps flow-map jumps (1 jump
        X_{0,1} while training with the CRPS objective). Differentiable.

        Returns:
        next_state: (B, N_grid, d_state)
        pred_std: None
        """
        n_steps = 1 if self.training else self._fmap_eval_steps()
        next_state = self._sample_next(
            prev_state,
            prev_prev_state,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            n_steps,
        )
        return next_state, None

    def _fmap_eval_steps(self):
        return getattr(self, "_fmap_steps_override", None) or self.fa(
            "fmap_sample_steps"
        )

    def predict_step_train(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
        target_state,
    ):
        """
        Flow-map loss on the true next state.

        Returns:
        next_state: (B, N_grid, d_state), a 1-jump sample (no gradient)
        loss: (B)
        """
        cond = (
            self.input_grid(prev_state, prev_prev_state, forcing),
            boundary_forcing,
            atmosphere_forcing,
        )
        x1 = self.state_to_target(target_state, prev_state)
        loss = self.flow_map_loss(x1, cond)
        with torch.no_grad():
            next_state = self._sample_next(
                prev_state,
                prev_prev_state,
                forcing,
                boundary_forcing,
                atmosphere_forcing,
                1,
            )
        return next_state, loss

    # ------------------------------------------------------------ training
    def _first_step_map_loss(self, batch):
        """Flow-map loss (B,) on the first step, from the true init states"""
        init_states, target_states, forcing, boundary_forcing = batch[:4]
        atmosphere_forcing = batch[4]
        if self.input_steps == 1:
            prev_state, prev_prev_state = init_states[:, 0], None
        else:
            prev_prev_state, prev_state = init_states[:, 0], init_states[:, 1]
        cond = (
            self.input_grid(prev_state, prev_prev_state, forcing[:, 0]),
            boundary_forcing[:, 0] if self.boundary_forced else None,
            atmosphere_forcing[:, 0] if self.atmosphere_forced else None,
        )
        x1 = self.state_to_target(target_states[:, 0], prev_state)
        return self.flow_map_loss(x1, cond)

    def crps_loss(self, batch):
        """Almost fair CRPS of --n_ens_train 1-jump rollouts, (B, pred_steps),
        as graph_crps"""
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            _,
        ) = batch
        trajectories = self.sample_trajectories(
            init_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            ensemble_size=self.fa("n_ens_train"),
        )  # (B, S, pred_steps, N, d_f)
        crps_batch = metrics.afcrps_ens(
            trajectories,
            target_states,
            None,
            average_grid=False,
            sum_vars=False,
            alpha=self.args.crps_alpha,
        )  # (B, pred_steps, N, d_f)
        return metrics.mask_and_reduce_metric(
            crps_batch / self.per_var_std,
            mask=self.loss_mask,
            average_grid=True,
            sum_vars=True,
        ), trajectories

    def training_step(self, batch):
        """Train on a single batch, with the current --fmap_objective"""
        if self.fmap_objective == "map":
            return super().training_step(batch)

        crps, trajectories = self.crps_loss(batch)
        crps = torch.mean(crps)
        log_dict = {"train_crps": crps}
        batch_loss = crps
        if self.fmap_objective == "map+crps":
            map_loss = torch.mean(self._first_step_map_loss(batch))
            log_dict["train_map_loss"] = map_loss
            batch_loss = crps + self.fa("fmap_crps_map_weight") * map_loss
        log_dict["train_loss"] = batch_loss
        with torch.no_grad():
            log_dict["train_mse"] = torch.mean(
                metrics.mse(
                    trajectories.mean(dim=1).detach(),
                    batch[1],
                    mask=self.loss_mask,
                )
            )
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    # ---------------------------------------------------------- validation
    def ensemble_step(self, batch):
        out = super().ensemble_step(batch)
        trajectories, target_states = out[0], out[1]
        if self._trainer is not None and not self._trainer.testing:
            crps = metrics.crps_ens(
                trajectories,
                target_states,
                None,
                mask=self.loss_mask,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.val_metrics["crps_ens"].append(crps)
            self.log(
                "val_crps_ens_mean",
                torch.mean(crps / self.diff_std),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch[0].shape[0],
            )
        return out

    def validation_step(self, batch, *args):
        super().validation_step(batch, *args)
        batch_idx = args[0]
        extra = self.fa("fmap_val_sample_steps_extra")
        if batch_idx == 0 and extra > 0:
            self._log_extra_step_metrics(batch, extra)

    @torch.no_grad()
    def _log_extra_step_metrics(self, batch, n_steps):
        """Ensemble CRPS and spread-skill with n_steps jumps, to compare with
        the default (--fmap_sample_steps) sampling"""
        init_states, target, forcing, boundary_forcing, atmosphere_forcing = (
            batch[:5]
        )
        self._fmap_steps_override = n_steps
        try:
            trajectories = self.sample_trajectories(
                init_states, forcing, boundary_forcing, atmosphere_forcing
            )
        finally:
            self._fmap_steps_override = None
        crps = metrics.crps_ens(
            trajectories, target, None, mask=self.loss_mask, sum_vars=False
        )
        spread_sq = metrics.spread_squared(
            trajectories, target, None, mask=self.loss_mask, sum_vars=False
        )
        ens_mse = metrics.mse(
            trajectories.mean(dim=1),
            target,
            None,
            mask=self.loss_mask,
            sum_vars=False,
        )
        m = self.ensemble_size
        spsk = math.sqrt((m + 1) / m) * (
            spread_sq.mean(dim=0).sqrt() / ens_mse.mean(dim=0).sqrt()
        )
        self.log_dict(
            {
                f"val_crps_ens_mean_{n_steps}steps": torch.mean(
                    crps / self.diff_std
                ),
                f"val_mean_spsk_ratio_{n_steps}steps": torch.mean(spsk),
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
