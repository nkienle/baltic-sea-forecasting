# Standard library
from typing import Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# First-party
from neural_lam import metrics
from neural_lam.models.ar_prob_model import ARProbModel
from neural_lam.models.graph_diff import GraphDiff

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore


class EDM(ARProbModel):
    """
    Diffusion Forecasting Model using the EDM framework.
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

        # ----------------------------------------------------------------------------
        # Diffusion (EDM) parameters
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.sigma_data = args.sigma_data
        self.rho = args.rho
        self.sampler = args.sampler
        self.sampler_steps = args.sampler_steps
        self.ensemble_size = args.ensemble_size
        self.var_leads_val_plot = args.var_leads_val_plot
        self.pred_residual = (
            args.pred_residual
        )  # Whether to predict the residual instead of the next state

        if args.backbone_model == "graph_diff":
            print("Using GraphDiff")
            self.model = GraphDiff(
                args,
                config,
                datastore,
                datastore_boundary,
                datastore_atmosphere,
            )
        else:
            raise NotImplementedError(
                f"Unknown backbone model: {args.backbone_model}"
            )

    # ----------------------------------------------------------------------------
    # EDM model methods
    def denoise(
        self,
        x,
        sigma,
        class_labels=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        **model_kwargs,
    ):
        """""
        Denoising forward pass through the diffusion backbone model.
        x: (B, N_grid, d_state)
        sigma: (B), noise level
        class_labels: (B, N_grid, d_conditioning)
        model_kwargs: additional arguments for the model

        Returns:
        D_x: (B, N_grid, d_state), denoised state
        """ ""
        sigma = sigma.reshape(-1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * x),
            c_noise.flatten(),
            class_labels,
            boundary_forcing=boundary_forcing,
            atmosphere_forcing=atmosphere_forcing,
            **model_kwargs,
        )
        D_x = c_skip * x + c_out * F_x

        return D_x

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
        """
        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)
        # (B, N_grid, d_input)

        latents = torch.randn_like(prev_state)  # (B, N_grid, d_state)

        # Run through sampler
        if self.sampler == "heun":
            next_state = self.heun_sampler(
                latents=latents,
                class_labels=input_grid,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
                sigma_min=self.sigma_min * 1.5,
                num_steps=self.sampler_steps,
            )
        elif self.sampler == "edm":
            next_state = self.edm_sampler(
                latents=latents,
                class_labels=input_grid,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
                sigma_min=self.sigma_min * 1.5,
                num_steps=self.sampler_steps,
            )

        # Add residual if needed
        if self.pred_residual:
            next_state = (
                next_state * self.diff_std
            ) + self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        # No predicted std.-dev. (ARModel.unroll_prediction contract)
        return next_state, None

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        Include metrics computation for ensemble mean prediction
        """

        target_states = batch[1]

        super().test_step(batch, batch_idx)

        (
            trajectories,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        ) = self.ensemble_step(batch)
        self.test_metrics["spread_squared"].append(spread_squared_batch)
        self.test_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(trajectories, dim=1)
        # (B, pred_steps, num_grid_nodes, d_f)

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            ens_std,
            mask=self.loss_mask,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            None,
            mask=self.loss_mask,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # super().on_test_epoch_end()
        # TODO: It would be nice if we can run this as well
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")
        self.log_spsk_ratio(self.test_metrics, "test")

    # Validation
    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        # TODO: Validation step (calculate loss), if it is meaningful?
        # Validation step batch 0, sample 1 trajectory for visual evaluation
        # Plot some example predictions using prior and encoder
        prediction, target, loss = self.common_step_train(batch)
        val_loss = torch.mean(loss)
        val_log_dict = {"val_loss": val_loss}
        # TODO: Calculate the validation loss
        batch_idx = args[0]
        if batch_idx == 0:
            # We only run the full validation for one batch
            # since sampling is expensive
            super().validation_step(batch, batch_idx)
            (
                trajectories,
                target_states,
                spread_squared_batch,
                ens_mse_batch,
            ) = self.ensemble_step(batch)
            self.val_metrics["spread_squared"].append(spread_squared_batch)
            self.val_metrics["ens_mse"].append(ens_mse_batch)

            if self.trainer.is_global_zero and self.n_example_pred > 0:
                # For now use val_steps_to_log to determine which steps
                # to make these plots for
                plot_log_steps = list(
                    filter(
                        lambda s: s <= target_states.shape[1],
                        self.args.val_steps_to_log,
                    )
                )
                # Plot forecasts
                traj_plots = self.plot_ensemble_examples(
                    batch,
                    n_examples=self.n_example_pred,
                    prediction=trajectories,
                    time_steps=plot_log_steps,
                    log=False,
                )

                # Store plots
                log_plot_dict = {}
                log_plot_dict.update(
                    {
                        f"prior_{plot_key}": plot
                        for plot_key, plot in traj_plots.items()
                    }
                )

                if not self.trainer.sanity_checking:
                    # Log all plots to wandb
                    wandb.log(log_plot_dict)

                plt.close("all")

        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

    # Training
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

        # Sample from F inverse
        rnd_uniform = torch.rand(
            [prev_state.shape[0], 1, 1], device=prev_state.device
        )
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max**rho_inv
        sigma_min_rho = self.sigma_min**rho_inv
        sigma = (
            sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho

        if prev_prev_state is not None:
            input_grid = torch.cat(
                (prev_state, prev_prev_state, forcing), dim=-1
            )
        else:
            input_grid = torch.cat((prev_state, forcing), dim=-1)

        # Make y residual if needed
        if self.pred_residual:
            y = target_state - prev_state
            y = (y - self.diff_mean) / self.diff_std  # Normalize residual
        else:
            y = target_state

        n = torch.randn_like(y) * sigma
        noisy_input = y + n

        next_state = self.denoise(
            noisy_input, sigma, input_grid, boundary_forcing, atmosphere_forcing
        )  # Shape (B, d_state, N_x, N_y)

        # Add residual if needed
        if self.pred_residual:
            next_state = (
                next_state * self.diff_std
            ) + self.diff_mean  # Unormalize residual
            next_state = prev_state + next_state

        # Calculate the loss
        # Weights for the loss function based on the noise level
        weight = (sigma**2 + self.sigma_data**2) / (
            sigma * self.sigma_data
        ) ** 2
        # (B)

        pred_std = self.per_var_std

        entry_mse = torch.nn.functional.mse_loss(
            next_state, target_state, reduction="none"
        )  # (..., N, d_state)
        entry_mse_weighted = entry_mse / (pred_std**2)  # (..., N, d_state)

        entry_mse_weighted = entry_mse_weighted * weight  # (B, N, d_state)

        loss = metrics.mask_and_reduce_metric(
            entry_mse_weighted,
            mask=self.loss_mask,
            average_grid=True,
            sum_vars=True,
        )

        return next_state, loss

    def unroll_prediction_train(
        self,
        init_states,
        forcing,
        boundary_forcing,
        atmosphere_forcing,
        target_states,
    ):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, input_steps, num_grid_nodes, d_f)
        forcing: (B, pred_steps, num_grid_nodes, d_static_f)
        boundary_forcing: (B, pred_steps, num_boundary_nodes, d_boundary_f)
        atmosphere_forcing:(
            B, pred_steps, num_atmosphere_nodes, d_atmosphere_f)
        """
        if self.input_steps == 1:
            prev_state = init_states[:, 0]
            prev_prev_state = None
        else:
            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1]
        prediction_list = []
        loss_list = []
        pred_steps = forcing.shape[1]

        for i in range(pred_steps):
            forcing_step = forcing[:, i]

            if self.boundary_forced:
                boundary_forcing_step = boundary_forcing[:, i]
            else:
                boundary_forcing_step = None

            if self.atmosphere_forced:
                atmosphere_forcing_step = atmosphere_forcing[:, i]
            else:
                atmosphere_forcing_step = None

            pred_state, loss = self.predict_step_train(
                prev_state,
                prev_prev_state,
                forcing_step,
                boundary_forcing_step,
                atmosphere_forcing_step,
                target_state=target_states[:, i],
            )
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            prediction_list.append(pred_state)
            loss_list.append(loss)

            # Update conditioning states
            if self.input_steps >= 2:
                prev_prev_state = prev_state
            prev_state = pred_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        loss_tensor = torch.stack(
            loss_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        return prediction, loss_tensor

    def common_step_train(self, batch):
        """
        Predict on single batch batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
        boundary_forcing:
            (B, pred_steps, num_boundary_nodes, d_boundary_forcing),
        atmosphere_forcing:
            (B, pred_steps, num_atmosphere_nodes, d_atmosphere_forcing),
        where index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            batch_times,
        ) = batch

        prediction, loss = self.unroll_prediction_train(
            init_states,
            forcing,
            boundary_forcing,
            atmosphere_forcing,
            target_states,
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, loss

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, loss = self.common_step_train(batch)

        # Compute loss
        batch_loss = torch.mean(loss)  # mean over unrolled times and batch

        batch_mse = torch.mean(
            metrics.mse(
                prediction,
                target,
                mask=self.loss_mask,
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss, "train_mse": batch_mse}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss

    # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    #
    # This work is licensed under a Creative Commons
    # Attribution-NonCommercial-ShareAlike 4.0 International License.
    # You should have received a copy of the license along with this
    # work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""

    # ----------------------------------------------------------------------------
    # Proposed EDM sampler (Algorithm 2).
    # TODO: Need to update samples with boundary forcing
    def edm_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        randn_like=torch.randn_like,
        num_steps=20,
        sigma_min=0.03,
        sigma_max=80,
        rho=7,
        S_churn=2.5,
        S_min=0.75,
        S_max=80,
        S_noise=1.05,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [
                torch.as_tensor(t_steps, device=latents.device),
                torch.zeros_like(t_steps[:1], device=latents.device),
            ]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next
            # diff_steps.append(x_cur)

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = torch.as_tensor(
                t_cur + gamma * t_cur, device=latents.device
            )
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * randn_like(x_cur, device=latents.device)

            # Euler step.
            denoised = self.denoise(
                x_hat,
                t_hat,
                class_labels=class_labels,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
            )
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x_next,
                    t_next,
                    class_labels=class_labels,
                    boundary_forcing=boundary_forcing,
                    atmosphere_forcing=atmosphere_forcing,
                )
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_next

    # ----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(
        self,
        latents,
        class_labels=None,
        boundary_forcing=None,
        atmosphere_forcing=None,
        num_steps=20,
        sigma_min=0.03,
        sigma_max=80,
        rho=7,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [
                torch.as_tensor(t_steps, device=latents.device),
                torch.zeros_like(t_steps[:1], device=latents.device),
            ]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            denoised = self.denoise(
                x_cur,
                t_cur,
                class_labels=class_labels,
                boundary_forcing=boundary_forcing,
                atmosphere_forcing=atmosphere_forcing,
            )
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x_next,
                    t_next,
                    class_labels=class_labels,
                    boundary_forcing=boundary_forcing,
                    atmosphere_forcing=atmosphere_forcing,
                )
                d_prime = (x_next - denoised) / t_next
                x_next = x_cur + (t_next - t_cur) * (
                    0.5 * d_cur + 0.5 * d_prime
                )

        return x_next
