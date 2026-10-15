# Standard library
from typing import Union

# Third-party
import torch

# First-party
from neural_lam.models.EDM import EDM

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class FM(EDM):
    """
    Flow Matching Forecasting Model.

    Linear interpolant z_t = (1 - t) z_0 + t z_1 between noise z_0 (t = 0) and
    the data z_1 (t = 1), the (normalized residual of the) next state. The
    backbone predicts the velocity b_t(z_t) = E[z_1 - z_0 | z_t].
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
        assert args.noise_embedding != "linear", (
            "FM needs a time embedding: use --noise_embedding positional "
            "(recommended, also for flow maps) or fourier. The linear "
            "embedding is for the noise vector of graph_crps."
        )

        # Clamp predictions to the valid range, as graph_crps/graph_det
        self.prepare_clamping_params(config, datastore)

    def input_grid(self, prev_state, prev_prev_state, forcing):
        """Conditioning features (B, N_grid, d_input)"""
        if prev_prev_state is not None:
            return torch.cat((prev_state, prev_prev_state, forcing), dim=-1)
        return torch.cat((prev_state, forcing), dim=-1)

    def state_to_target(self, target_state, prev_state):
        """The generated variable z_1: the normalized residual if
        --pred_residual, else the state itself"""
        if self.pred_residual:
            y = target_state - prev_state
            return (y - self.diff_mean) / self.diff_std
        return target_state

    def target_to_state(self, y, prev_state):
        """Inverse of state_to_target, with output clamping"""
        if self.pred_residual:
            delta = y * self.diff_std + self.diff_mean
            return self.get_clamped_new_state(delta, prev_state)
        return y

    def velocity_net(
        self, z, t, input_grid, boundary_forcing, atmosphere_forcing
    ):
        """
        Velocity b_t(z) from the backbone
        z: (B, N_grid, d_state)
        t: (B,) or scalar time
        """
        t = torch.as_tensor(t, device=z.device, dtype=z.dtype)
        t = t.reshape(-1).expand(z.shape[0])
        return self.model(
            z, t, input_grid, boundary_forcing, atmosphere_forcing
        )

    # Evaluation
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
        pred_std: None
        """
        input_grid = self.input_grid(prev_state, prev_prev_state, forcing)
        # (B, N_grid, d_input)

        latents = torch.randn_like(prev_state)  # (B, N_grid, d_state)

        # Run through sampler
        if self.sampler == "heun":
            y = self.heun_sampler(
                latents=latents,
                class_labels=input_grid,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
                num_steps=self.sampler_steps,
            )
        elif self.sampler == "stochastic":
            y = self.stochastic_sampler(
                latents=latents,
                class_labels=input_grid,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
                num_steps=self.sampler_steps,
            )
        else:
            raise ValueError(f"Unknown FM sampler: {self.sampler}")

        return self.target_to_state(y, prev_state), None

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
        forcing: (B, N_grid, forcing_dim), dynamic forcing
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        atmosphere_forcing: (B, num_atmosphere_nodes, atmosphere_forcing_dim)
        target_state: (B, N_grid, d_state), true state X_{t+1} at time t+1

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at t+1
        loss: (B)
        """
        input_grid = self.input_grid(prev_state, prev_prev_state, forcing)

        z1 = self.state_to_target(target_state, prev_state)
        z0 = torch.randn_like(z1)
        t = torch.rand([prev_state.shape[0], 1, 1], device=prev_state.device)
        zt = (1 - t) * z0 + t * z1

        pred_drift = self.velocity_net(
            zt, t.flatten(), input_grid, boundary_forcing, atmosphere_forcing
        )

        # This predicts the drift b
        drift = z1 - z0

        # z is in normalized-residual units, so only the feature weights
        # remain (per_var_std = diff_std / sqrt(w))
        pred_std = self.per_var_std / self.diff_std if self.pred_residual \
            else self.per_var_std

        loss = self.loss(
            pred_drift,
            drift,
            pred_std,
            mask=self.loss_mask,
        )  # (B)

        # This is the predicted E[z1 | zt]
        next_state = self.target_to_state(
            zt + pred_drift * (1 - t), prev_state
        )

        return next_state, loss

    # ----------------------------------------------------------------------------
    # Sampling methods below adapted from:
    # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    #
    # This work is licensed under a Creative Commons
    # Attribution-NonCommercial-ShareAlike 4.0 International License.
    # You should have received a copy of the license along with this
    # work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""

    # ----------------------------------------------------------------------------
    # Heun sampler on the probability flow ODE, num_steps steps from t=0 to 1
    def heun_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        num_steps=20,
    ):
        t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)

        # Main sampling loop.
        x_next = latents
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Euler step.
            x_cur = x_next
            d_cur = self.velocity_net(
                x_cur, t_cur, class_labels, boundary_forcing, atmosphere_forcing
            )
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction (not on the last step)
            if i < num_steps - 1:
                d_prime = self.velocity_net(
                    x_next,
                    t_next,
                    class_labels,
                    boundary_forcing,
                    atmosphere_forcing,
                )
                x_next = x_cur + (t_next - t_cur) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_next

    # Euler-Maruyama on the SDE with the same marginals, diffusion eps*(1-t)
    def stochastic_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        num_steps=100,
    ):
        tmin = 0.0
        tmax = 1.0
        eps = 1.0

        # Time step discretization.
        ts = torch.linspace(tmin, tmax, num_steps + 1, device=self.device)
        dt = (tmax - tmin) / num_steps

        # Main sampling loop.
        zt = latents  # Initialize with noise
        for t in ts[:-1]:
            alpha_t = t
            beta_t = 1 - t
            gamma_t = 1
            alpha_dot_t = 1
            # beta_dot_t = -1
            eps_t = eps * beta_t

            b = self.velocity_net(
                zt, t, class_labels, boundary_forcing, atmosphere_forcing
            )
            s = (alpha_t * b - alpha_dot_t * zt) / (
                beta_t * gamma_t
            )  # s = (t * b - zt) / (1 - t)
            dz = b + eps_t * s

            dW = torch.randn_like(zt) * torch.sqrt(2 * dt * eps_t)

            zt = zt + dz * dt + dW

        return zt  # (B, N_grid, d_state)
