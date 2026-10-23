"""
Tests of the graph_crps sources of stochasticity (--stochasticity fgn / mc /
sdl), the noise ramp and the loading of graph_det checkpoints.

The bsf env has no pytest, so this also runs as a script (from the repo root):
    python tests/test_graph_crps_stochasticity.py         # all tests
    python tests/test_graph_crps_stochasticity.py unit    # only the ones
                                                          # without data
The model tests build small graph_det / graph_crps models from the 1 degree
config used by the training scripts and take one batch (slow on CPU).
"""

# Standard library
import argparse
import functools
import itertools
import sys
from pathlib import Path

# Third-party
import torch

# First-party
import neural_lam.train_model as train_model
from neural_lam import metrics
from neural_lam.models.graph_diff import StochasticLevelNoise
from neural_lam.utils import make_mlp, set_mc_dropout

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
]
SOURCE_COMBINATIONS = [
    list(c)
    for n in (1, 2, 3)
    for c in itertools.combinations(("fgn", "mc", "sdl"), n)
]


# ----------------------------------------------------------------------------
# Unit tests (no data)
def test_mc_dropout_keeps_parameter_layout():
    for cond_dim in (None, 4):
        mlp = make_mlp([8, 16, 16], cond_dim=cond_dim)
        keys = list(mlp.state_dict())
        set_mc_dropout(mlp, 0.5)
        assert list(mlp.state_dict()) == keys


def test_mc_dropout_off_is_identity_and_on_in_eval():
    for cond_dim in (None, 4):
        mlp = make_mlp([8, 16, 16], cond_dim=cond_dim).eval()
        x, emb = torch.randn(3, 5, 8), torch.randn(3, 1, 4)
        reference = mlp(x, emb)
        set_mc_dropout(mlp, 0.0)
        assert torch.equal(mlp(x, emb), reference)
        set_mc_dropout(mlp, 0.5)
        # Active also in eval mode (MC dropout), so two calls differ
        assert not torch.equal(mlp(x, emb), mlp(x, emb))


def test_mc_dropout_skips_mlps_without_hidden_layer():
    mlp = make_mlp([8, 16], cond_dim=4)
    x, emb = torch.randn(3, 5, 8), torch.randn(3, 1, 4)
    reference = mlp(x, emb)
    set_mc_dropout(mlp, 0.5)
    assert torch.equal(mlp(x, emb), reference)


def test_level_noise_scale_and_resampling():
    sdl = StochasticLevelNoise(16, noise_dim=4, alpha=0.5)
    x = torch.randn(2, 7, 16)
    assert torch.equal(sdl(x, scale=0.0), x)
    assert not torch.equal(sdl(x), sdl(x))
    # Same RNG state, same noise (what gradient checkpointing relies on)
    torch.manual_seed(3)
    a = sdl(x)
    torch.manual_seed(3)
    assert torch.equal(sdl(x), a)


# ----------------------------------------------------------------------------
# Model tests (small graph_det / graph_crps on the 1 degree data)
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
    """(parser, model kwargs, batch with 2 AR steps), built once"""
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
    args = parser.parse_args(COMMON_ARGS + ["--model", "graph_det"])
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
        batch_size=1,
        num_workers=1,
        use_atmosphere_g2m=args.use_atmosphere_g2m,
        density_channel=getattr(config.training, "density_channel", None),
    )
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    # Collate in-process (a worker process is not needed for 1 sample)
    batch = loader.collate_fn([data_module.train_dataset[0]])
    return parser, kwargs, batch


def _make(model_name, extra_args=()):
    parser, kwargs, _ = _setup()
    args = parser.parse_args(
        COMMON_ARGS + ["--model", model_name] + list(extra_args)
    )
    return train_model.MODELS[model_name](args, **kwargs)


@functools.lru_cache(maxsize=None)
def _graph_det():
    torch.manual_seed(1)
    return _make("graph_det")


def _crps_from_det(sources, extra_args=()):
    """graph_crps with the given sources, loaded from the graph_det weights
    through on_load_checkpoint (as trainer.fit(ckpt_path=...) does)"""
    crps = _make(
        "graph_crps", ["--stochasticity", *sources, *extra_args]
    )
    checkpoint = {"state_dict": dict(_graph_det().state_dict())}
    crps.on_load_checkpoint(checkpoint)
    crps.load_state_dict(checkpoint["state_dict"])  # strict
    return crps


def _step_args(batch):
    init, _, forcing, boundary_forcing, atmosphere_forcing, _ = batch
    return (
        init[:, 1],
        init[:, 0],
        forcing[:, 0],
        None if boundary_forcing is None else boundary_forcing[:, 0],
        atmosphere_forcing[:, 0],
    )


def test_graph_det_checkpoint_loads_and_only_sdl_is_new():
    crps = _make("graph_crps", ["--stochasticity", "sdl"])
    det_keys = set(_graph_det().state_dict())
    crps_keys = set(crps.state_dict())
    assert det_keys <= crps_keys
    assert all(k.startswith("model.sdl.") for k in crps_keys - det_keys)
    assert crps_keys - det_keys
    _crps_from_det(["sdl"])  # Does not raise
    # Without sdl the parameter layout is exactly that of graph_det
    assert set(_make("graph_crps", ["--stochasticity", "mc"]).state_dict()) == (
        det_keys
    )


def test_zero_ramp_is_graph_det():
    _, _, batch = _setup()
    det = _graph_det().train()
    with torch.no_grad():
        reference, _ = det.predict_step(*_step_args(batch))
    for sources in SOURCE_COMBINATIONS:
        crps = _crps_from_det(sources).train()
        crps._train_stoch_ramp = 0.0
        with torch.no_grad():
            pred, _ = crps.predict_step(*_step_args(batch))
        assert torch.equal(pred, reference), sources


def test_each_source_gives_spread():
    _, _, batch = _setup()
    for sources in SOURCE_COMBINATIONS:
        crps = _crps_from_det(sources).eval()  # eval: ramp 1, MC dropout on
        with torch.no_grad():
            a, _ = crps.predict_step(*_step_args(batch))
            b, _ = crps.predict_step(*_step_args(batch))
        assert (a - b).abs().max() > 0, sources


def test_no_source_gives_no_spread():
    _, _, batch = _setup()
    crps = _crps_from_det([]).eval()
    with torch.no_grad():
        a, _ = crps.predict_step(*_step_args(batch))
        b, _ = crps.predict_step(*_step_args(batch))
    assert torch.equal(a, b)


def test_gradients_same_with_checkpointing():
    """All noise must be identical in the checkpoint recomputation"""
    _, _, batch = _setup()
    init, target, forcing, boundary_forcing, atmosphere_forcing, _ = batch
    for sources in SOURCE_COMBINATIONS:
        crps = _crps_from_det(sources).train()
        crps._train_stoch_ramp = 0.7
        grads = []
        for checkpointing in (False, True):
            crps.grad_checkpointing = checkpointing
            crps.zero_grad()
            torch.manual_seed(5)
            trajectories = crps.sample_trajectories(
                init, forcing, boundary_forcing, atmosphere_forcing,
                ensemble_size=2,
            )
            loss = metrics.afcrps_ens(
                trajectories, target, None, average_grid=False,
                sum_vars=False, alpha=0.95,
            ).mean()
            loss.backward()
            grads.append(
                torch.cat(
                    [
                        p.grad.flatten()
                        for p in crps.parameters()
                        if p.grad is not None
                    ]
                )
            )
        assert grads[0].shape == grads[1].shape, sources
        assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6), (
            sources,
            (grads[0] - grads[1]).abs().max(),
        )


def test_sdl_parameters_get_gradients():
    _, _, batch = _setup()
    init, target, forcing, boundary_forcing, atmosphere_forcing, _ = batch
    for points in ("down", "init"):
        crps = _crps_from_det(["sdl"], ["--sdl_points", points]).train()
        trajectories = crps.sample_trajectories(
            init, forcing, boundary_forcing, atmosphere_forcing,
            ensemble_size=2,
        )
        metrics.afcrps_ens(
            trajectories, target, None, average_grid=False, sum_vars=False
        ).mean().backward()
        sdl_grads = [
            p.grad for n, p in crps.named_parameters() if ".sdl." in n
        ]
        assert sdl_grads and all(
            g is not None and g.abs().sum() > 0 for g in sdl_grads
        ), points


def test_sdl_levels_subset():
    crps = _make(
        "graph_crps", ["--stochasticity", "sdl", "--sdl_levels", "1", "2"]
    )
    assert all(set(d.keys()) == {"1", "2"} for d in crps.model.sdl)
    assert len(crps.model.sdl) == crps.args.processor_layers


def test_mc_dropout_scope():
    counts = {
        scope: len(
            _make(
                "graph_crps",
                ["--stochasticity", "mc", "--mc_dropout_scope", scope],
            )._mc_dropout_mlps
        )
        for scope in ("processor", "mesh", "all")
    }
    assert 0 < counts["processor"] < counts["mesh"] < counts["all"], counts


if __name__ == "__main__":
    only_unit = len(sys.argv) > 1 and sys.argv[1] == "unit"
    tests = [
        (name, fn)
        for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    unit = {
        "test_mc_dropout_keeps_parameter_layout",
        "test_mc_dropout_off_is_identity_and_on_in_eval",
        "test_mc_dropout_skips_mlps_without_hidden_layer",
        "test_level_noise_scale_and_resampling",
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
