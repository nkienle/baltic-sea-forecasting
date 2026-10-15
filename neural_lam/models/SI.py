# Standard library
from typing import Union

# Third-party
import torch
import torch.nn.functional as F

# First-party
from neural_lam.models.EDM import EDM

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))


class SI(EDM):
    """
    Stochastic Interpolants Forecasting Model.
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
        self.GT = None
        self.sampler = "euler"  # TODO: Only euler for now
        # TODO: Should be able to change sigma_coef
        self.interpolant = Interpolant(sigma_coef=1, beta_fn="t^2")
        self.t_min_sampling = 0.0  # no min time needed
        self.t_max_sampling = 0.999

    # Evaluation
    def EM(
        self,
        base=None,
        cond=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        diffusion_fn=None,
    ):
        steps = self.sampler_steps
        tmin, tmax = self.t_min_sampling, self.t_max_sampling
        ts = torch.linspace(tmin, tmax, steps).type_as(base)
        dt = ts[1] - ts[0]
        ones = torch.ones(base.shape[0]).type_as(base)

        # initial condition
        xt = base

        # diffusion_fn = None means use the diffusion function that you
        # trained with. Otherwise, for a desired diffusion coefficient,
        # do the model surgery to define the correct drift coefficient

        def step_fn(xt, t):
            D = self.interpolant.interpolant_coefs(
                {"t": t, "zt": xt, "z0": base}
            )

            bF = self.model(xt, t, cond, boundary_forcing, atmosphere_forcing)
            D["bF"] = bF
            sigma = self.interpolant.sigma(t)

            # specified diffusion func
            if diffusion_fn is not None:
                g = diffusion_fn(t)
                s = self.drift_to_score(D)
                f = bF + 0.5 * (g.pow(2) - sigma.pow(2)) * s

            # default diffusion func
            else:
                f = bF
                g = sigma

            mu = xt + f * dt
            xt = mu + g * torch.randn_like(mu) * dt.sqrt()
            return xt, mu  # return sample and its mean

        # TODO: Only supporting the same diffusion function for now.
        def step_fn_2(xt, t, t1):
            bF = self.model(xt, t, cond, boundary_forcing, atmosphere_forcing)

            mu1 = xt + bF * dt

            bF2 = self.model(
                mu1, t1, cond, boundary_forcing, atmosphere_forcing
            )

            f = 0.5 * (bF + bF2)

            # Final step
            sigma = self.interpolant.sigma(t)
            g = sigma
            mu = xt + f * dt
            xt = mu + g * torch.randn_like(mu) * dt.sqrt()
            return xt, mu  # return sample and its mean

        for i, tscalar in enumerate(ts):

            if i == 0 and (diffusion_fn is not None):
                # only need to do this when using other diffusion coefficients
                # that you didn't train with because the drift-to-score
                # conversion has a denominator that features 0 at time 0
                # if just sampling with "sigma" (the diffusion coefficient
                # you trained with) you can skip this
                tscalar = ts[1]  # 0 + (1/500)

            if self.sampler == "euler_2" and i < len(ts) - 1:
                xt, mu = step_fn_2(xt, tscalar * ones, ts[i + 1] * ones)
            else:
                xt, mu = step_fn(xt, tscalar * ones)

        assert not bad(mu)
        return mu

    # TODO: Should support going from pure noise to the reference
    # distribution to compare with diffusion.
    def predict_step(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)


        Returns:
        next_state: (B, N_grid, d_state),
            predicted weather state X_{t+1} at time t+1
        """
        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)
        # (B, N_grid, d_input)

        # definitely_sample
        EM_args = {
            "base": prev_state,
            "cond": input_grid,
            "boundary_forcing": boundary_forcing,
            "atmosphere_forcing": atmosphere_forcing,
        }

        # list diffusion funcs
        # None means use the one you trained with
        # diffusion_fns = {
        #     "g_sigma": None,
        #     "g_sigma_01": lambda t: self.sigma_coef * self.wide(1 - t) * 0.1,
        #     "g_other": lambda t: self.sigma_coef * self.wide(1 - t).pow(4),
        # }

        # None because we want to use the diffusion function we trained with,
        # TODO: Experiment with this later
        next_state = self.EM(diffusion_fn=None, **EM_args)

        # No predicted std.-dev. (ARModel.unroll_prediction contract)
        return next_state, None

    # TODO: Should support going from pure noise to the reference
    # distribution to compare with diffusion.
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
        Predict weather state one time step ahead
        X_{t-1}, X_t -> X_t+1

        prev_state: (B, N_grid, d_state), weather state X_t at time t
        prev_prev_state: (B, N_grid, d_state), weather state X_{t-1} at time t-1
        batch_static_features: (B, N_grid, batch_static_feature_dim), static
        forcing: (B, N_grid, forcing_dim), dynamic forcing
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        target_state: (B, N_grid, d_state), true state X_{t+1} at time t+1

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at t+1
        loss: (B)
        """
        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)

        # Prepare batch
        D = {
            "z0": prev_state,
            "z1": target_state,
            "label": None,
            "N": prev_state.shape[0],
        }

        # Get random batch of times
        D["t"] = torch.rand(prev_state.shape[0], device=prev_state.device)

        # Interpolant noise
        D["noise"] = torch.randn_like(prev_state, device=prev_state.device)

        # Get alpha, beta, etc
        D = self.interpolant.interpolant_coefs(D)

        # zt
        D["zt"] = self.interpolant.compute_zt(D)

        # Target
        D["drift_target"] = self.interpolant.compute_target(D)

        output = self.model(
            D["zt"],
            D["t"].reshape(D["zt"].shape[0]),
            input_grid,
            boundary_forcing,
            atmosphere_forcing,
        )  # Shape (B, d_state, N_x, N_y)

        # Calculate loss
        loss = F.mse_loss(output, D["drift_target"], reduction="none")

        loss = loss / (self.per_var_std**2)

        return output, loss


class Interpolant:

    def __init__(self, sigma_coef=1, beta_fn="t^2"):
        self.sigma_coef = sigma_coef
        self.beta_fn = beta_fn

    def wide(self, t):
        return t[:, None, None]

    def alpha(self, t):
        return self.wide(1 - t)

    def alpha_dot(self, t):
        return self.wide(-1.0 * torch.ones_like(t))

    def beta(self, t):
        is_squared = self.beta_fn == "t^2"
        return self.wide(t.pow(2) if is_squared else t)

    def beta_dot(self, t):
        is_squared = self.beta_fn == "t^2"
        return self.wide(2.0 * t if is_squared else torch.ones_like(t))

    # we sometimes multiply sigma + sigma_dot by avg pixel norm,
    # but when standardized (centered cifar),
    # or when norm 1 (we rescale nse), not needed
    def sigma(self, t):
        return self.sigma_coef * self.wide(1 - t)

    def sigma_dot(self, t):
        return self.sigma_coef * self.wide(-torch.ones_like(t))

    def gamma(self, t):
        return self.wide(t.sqrt()) * self.sigma(t)

    def compute_zt(self, D):
        return D["at"] * D["z0"] + D["bt"] * D["z1"] + D["gamma_t"] * D["noise"]

    def compute_target(self, D):
        return (
            D["adot"] * D["z0"]
            + D["bdot"] * D["z1"]
            + (D["sdot"] * D["root_t"]) * D["noise"]
        )

    def interpolant_coefs(self, D):
        return self(D)

    def __call__(self, D):
        D["at"] = self.alpha(D["t"])
        D["bt"] = self.beta(D["t"])
        D["adot"] = self.alpha_dot(D["t"])
        D["bdot"] = self.beta_dot(D["t"])
        D["root_t"] = self.wide(D["t"].sqrt())
        D["gamma_t"] = self.gamma(D["t"])
        D["st"] = self.sigma(D["t"])
        D["sdot"] = self.sigma_dot(D["t"])
        return D
