"""
Tests of the FM / FlowMap models and the backbone changes they rely on.

The bsf env has no pytest, so this also runs as a script (from the repo root):
    python tests/test_flow_map.py            # all tests
    python tests/test_flow_map.py unit       # only the ones without data
The model tests build a small FM and FlowMap from the 1 degree config used by
the training scripts and take one training batch (slow on CPU, minutes).
"""

# Standard library
import argparse
import functools
import sys
from pathlib import Path

# Third-party
import torch

# First-party
import neural_lam.train_model as train_model
from neural_lam.models.flow_map import LogVarWeight
from neural_lam.models.graph_diff import DtEmbedding
from neural_lam.utils import ConditionalLayerNorm, set_adaln_zero

CONFIG_PATH = Path("data/global_ocean_1_density.yaml")
GRAPH_NAME = "global_cluster_1_deg_20_refinement_4_levels"
COMMON_ARGS = [
    "--config_path", str(CONFIG_PATH),
    "--graph", GRAPH_NAME,
    "--hidden_dim", "64",
    "--hidden_dim_grid", "32",
    "--hidden_dim_mesh_nodes", "32",
    "--hidden_dim_edge", "16",
    "--processor_layers", "2",
    "--input_steps", "2",
    "--batch_size", "1",
    "--noise_embedding", "positional",
    "--pred_residual",
]


# ----------------------------------------------------------------------------
# Unit tests (no data)
def test_adaln_zero_is_identity_at_init():
    norm = ConditionalLayerNorm(8, cond_dim=4)
    set_adaln_zero(norm)
    x, emb = torch.randn(3, 5, 8), torch.randn(3, 1, 4)
    expected = torch.nn.functional.layer_norm(x, (8,))
    assert torch.allclose(norm(x, emb), expected, atol=1e-6)


def test_adaln_zero_keeps_parameter_layout():
    plain, zero = ConditionalLayerNorm(8, 4), ConditionalLayerNorm(8, 4)
    set_adaln_zero(zero)
    assert {k: v.shape for k, v in plain.state_dict().items()} == {
        k: v.shape for k, v in zero.state_dict().items()
    }


def test_dt_embedding_is_zero_at_init():
    emb = DtEmbedding(16)
    assert torch.all(emb(torch.rand(5)) == 0)


def test_logvar_weight_is_bounded():
    w = LogVarWeight()
    with torch.no_grad():
        w.weight.mul_(100.0)  # Scale is removed by the normalisation
    s, t = torch.rand(1000), torch.rand(1000)
    assert w(s, t).abs().max() <= 2**0.5 + 1e-5


# ----------------------------------------------------------------------------
# Model tests (small FM / FlowMap on the 1 degree data)
def _build_arg_parser():
    """The real train_model parser, captured on its way to parse_args"""
    captured = {}
    original_parse_args = argparse.ArgumentParser.parse_args

    class _Captured(Exception):
        pass

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _Captured()

    argparse.ArgumentParser.parse_args = _capture
    try:
        train_model.main([])
    except _Captured:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
    return captured["parser"]


@functools.lru_cache(maxsize=None)
def _setup():
    """(fm, flow_map, batch), built once"""
    torch.manual_seed(0)
    parser = _build_arg_parser()
    (
        config,
        datastore,
        datastore_boundary,
        datastore_atmosphere,
        statistics_datastore,
        statistics_datastore_boundary,
        statistics_datastore_atmosphere,
    ) = train_model.load_config_and_datastores(config_path=str(CONFIG_PATH))
    kwargs = dict(
        config=config,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        statistics_datastore=statistics_datastore,
        statistics_datastore_boundary=statistics_datastore_boundary,
        statistics_datastore_atmosphere=statistics_datastore_atmosphere,
    )
    fm = train_model.MODELS["FM"](
        parser.parse_args(COMMON_ARGS + ["--model", "FM"]), **kwargs
    )
    args = parser.parse_args(COMMON_ARGS + ["--model", "FlowMap", "--fmap_ema"])
    fmap = train_model.MODELS["FlowMap"](args, **kwargs)

    data_module = train_model.WeatherDataModule(
        **{k: v for k, v in kwargs.items() if k != "config"},
        ar_steps_train=2,
        ar_steps_eval=2,
        input_steps=2,
        standardize=True,
        num_past_forcing_steps=args.num_past_forcing_steps,
        num_future_forcing_steps=args.num_future_forcing_steps,
        current_forcing_step=bool(args.current_forcing_step),
        current_boundary_step=bool(args.current_boundary_step),
        current_atmosphere_step=bool(args.current_atmosphere_step),
        num_past_boundary_steps=args.num_past_boundary_steps,
        num_future_boundary_steps=args.num_future_boundary_steps,
        num_past_atmosphere_steps=args.num_past_atmosphere_steps,
        num_future_atmosphere_steps=args.num_future_atmosphere_steps,
        batch_size=2,
        num_workers=1,
        use_atmosphere_g2m=args.use_atmosphere_g2m,
        density_channel=getattr(config.training, "density_channel", None),
    )
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    # Collate in-process (a worker process is not needed for 2 samples)
    batch = loader.collate_fn(
        [data_module.train_dataset[i] for i in range(2)]
    )
    return fm, fmap, batch


def _cond(model, batch):
    init, _, forcing, _, atmosphere_forcing, _ = batch
    return (
        model.input_grid(init[:, 1], init[:, 0], forcing[:, 0]),
        None,
        atmosphere_forcing[:, 0],
    )


def test_fm_loss_and_gradients_finite():
    fm, _, batch = _setup()
    init, target, forcing, _, atmosphere_forcing, _ = batch
    fm.train()
    fm.zero_grad()
    _, loss = fm.predict_step_train(
        init[:, 1], init[:, 0], forcing[:, 0], None,
        atmosphere_forcing[:, 0], target[:, 0],
    )
    loss.mean().backward()
    assert torch.isfinite(loss).all()
    assert all(
        torch.isfinite(p.grad).all()
        for p in fm.parameters()
        if p.grad is not None
    )


def test_flow_map_from_teacher_is_fm_velocity():
    fm, fmap, batch = _setup()
    fmap._init_from_teacher(fm)
    with torch.no_grad():
        fmap.model.map_dt.proj.weight.zero_()
        fmap.model.map_dt.proj.bias.zero_()
    cond = _cond(fmap, batch)
    x = torch.randn_like(batch[1][:, 0])
    s, t = torch.tensor([0.2, 0.6]), torch.tensor([0.7, 0.9])
    with torch.no_grad():
        phi = fmap._phi(s, t, x, cond)
        b = fm.velocity_net(x, s, *cond)
    assert torch.allclose(phi, b, atol=1e-6)


def test_jvp_matches_finite_differences():
    _, fmap, batch = _setup()
    cond = _cond(fmap, batch)
    with torch.no_grad():
        fmap.model.map_dt.proj.weight.normal_(0, 0.1)
    x = torch.randn_like(batch[1][:, 0])
    s, t = torch.tensor([0.2, 0.6]), torch.tensor([0.7, 0.9])

    def f(tt):
        return fmap.flow_map(s, tt, x, cond)

    _, dX = torch.func.jvp(f, (t,), (torch.ones_like(t),))
    h = 1e-3
    with torch.no_grad():
        fd = (f(t + h) - f(t - h)) / (2 * h)
    assert ((dX - fd).norm() / fd.norm()) < 1e-2


def test_map_losses_finite():
    _, fmap, batch = _setup()
    fmap.train()
    init, target = batch[0], batch[1]
    x1 = fmap.state_to_target(target[:, 0], init[:, 1])
    cond = _cond(fmap, batch)
    for loss_name in ("lsd", "psd"):
        fmap.args.fmap_loss = loss_name
        fmap.setup_flow_map()
        fmap.zero_grad()
        loss = fmap.flow_map_loss(x1, cond)
        loss.mean().backward()
        assert torch.isfinite(loss).all(), loss_name
        assert fmap.model.map_dt.proj.weight.grad.abs().sum() > 0
        assert fmap.fmap_logvar.weight.grad.abs().sum() > 0
    fmap.args.fmap_loss = "lsd"
    fmap.setup_flow_map()


def test_crps_gradients_same_with_checkpointing():
    """The rollout noise must be identical in the checkpoint recomputation"""
    _, fmap, batch = _setup()
    fmap.train()
    grads = []
    for checkpointing in (False, True):
        fmap.grad_checkpointing = checkpointing
        fmap.zero_grad()
        torch.manual_seed(1)
        crps, _ = fmap.crps_loss(batch)
        crps.mean().backward()
        grads.append(fmap.model.output_map.mlp_layers[-1].weight.grad.clone())
    fmap.grad_checkpointing = False
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_ema_swap_roundtrip():
    _, fmap, _ = _setup()
    fmap.reset_ema()
    with torch.no_grad():
        for p in fmap.model.parameters():
            p.add_(1.0)
    raw = [p.clone() for p in fmap._ema_params()]
    fmap.swap_in_ema()
    assert any(
        not torch.equal(p, r) for p, r in zip(fmap._ema_params(), raw)
    )
    fmap.swap_out_ema()
    assert all(torch.equal(p, r) for p, r in zip(fmap._ema_params(), raw))


if __name__ == "__main__":
    only_unit = len(sys.argv) > 1 and sys.argv[1] == "unit"
    tests = [
        (name, fn)
        for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    unit = {
        "test_adaln_zero_is_identity_at_init",
        "test_adaln_zero_keeps_parameter_layout",
        "test_dt_embedding_is_zero_at_init",
        "test_logvar_weight_is_bounded",
    }
    failed = 0
    for name, fn in tests:
        if only_unit and name not in unit:
            continue
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as err:  # pylint: disable=broad-except
            failed += 1
            print(f"FAIL {name}: {type(err).__name__}: {err}", flush=True)
    sys.exit(1 if failed else 0)
