# Standard library
from typing import Union

# Third-party
import numpy as np
import torch
from torch import nn
from torch.nn.functional import silu

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..interaction_net import FlexibleNet, InteractionNet, PropagationNet
from ..utils import make_mlp
from .ar_model import ARModel


class GraphDiff(ARModel):
    """
    Hierarchical Graph-based Diffusion Forecasting Model
    with message passing that goes sequentially down
    and up the hierarchy during processing.
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
            config=config,
            datastore=datastore,
            datastore_boundary=datastore_boundary,
            datastore_atmosphere=datastore_atmosphere,
            **kwargs,
        )
        print("Initializing GraphDiff")

        self.channel_mult_emb = args.channel_mult_emb
        self.channel_mult_noise = args.channel_mult_noise
        self.remove_cond = True if args.model == "SI" else False

        # Use the density-aware count set by ARModel.__init__ (includes the
        # density channel when config.training.density_channel is set), so the
        # output map and noise-channel input match the N+1 target/loss dims.
        num_state_vars = self.num_state_vars

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape

        # NOTE: grid_output_dim is inherited from ARModel, which sets it to
        # num_state_vars, or 2*num_state_vars when --output_std is given. The
        # diffusion-style users of this backbone (EDM/FM/SI/graph_crps) never
        # pass --output_std, so for them this is num_state_vars, i.e. just the
        # denoised state. graph_det follows graph_fm and supports both.

        # Add the noise channel for the diffusion process
        if args.noise_embedding != "linear":
            self.interior_input_dim += num_state_vars

        # ----------------------------------------------------------------------------
        # BaseGraphModel parameters
        assert (
            args.eval is None or args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size during validation"

        # Load graph with static features
        # NOTE: (IMPORTANT!) mesh nodes MUST have the first
        # num_mesh_nodes indices,
        graph_dir_path = datastore.root_path / "graphs" / args.graph
        self.hierarchical, graph_ldict = utils.load_graph(
            graph_dir_path=graph_dir_path,
            datastore=datastore,
        )
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        utils.rank_zero_print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # Determine grid hidden dim
        if args.hidden_dim_grid is None:
            hidden_dim_grid = args.hidden_dim
        else:
            hidden_dim_grid = args.hidden_dim_grid

        # Determine edge hidden dim
        if args.hidden_dim_edge is None:
            hidden_dim_edge = hidden_dim_grid
        else:
            hidden_dim_edge = args.hidden_dim_edge

        # Determine mesh node hidden dim
        if args.hidden_dim_mesh_nodes is None:
            hidden_dim_mesh_nodes = args.hidden_dim
        else:
            hidden_dim_mesh_nodes = args.hidden_dim_mesh_nodes

        print(
            f"Using hidden_dim_grid={hidden_dim_grid}, "
            f"hidden_dim_mesh_nodes={hidden_dim_mesh_nodes}, "
            f"hidden_dim_edge={hidden_dim_edge}, hidden_dim={args.hidden_dim}"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_mlp_blueprint_end = [hidden_dim_grid] * (
            args.hidden_layers + 1
        )
        self.edge_mlp_blueprint_end = [hidden_dim_edge] * (
            args.hidden_layers + 1
        )

        print(f"Using noise embedding: {args.noise_embedding}")
        self.noise_level_dim = (
            self.mlp_blueprint_end[-1] * self.channel_mult_emb
        )
        if args.noise_embedding == "linear":
            self.noise_dim = args.noise_dim
            self.map_noise = LinearNoiseEmbedding(
                noise_dim=self.noise_dim,
                hidden_dim=self.noise_level_dim,
                output_dim=self.noise_level_dim,
            )
        elif args.noise_embedding == "positional":
            self.noise_dim = 16
            self.map_noise = PositionalNoiseEmbedding(
                num_frequencies=args.hidden_dim * self.channel_mult_noise,
                hidden_dim=self.noise_level_dim,
                output_dim=self.noise_level_dim,
            )
        else:
            self.noise_dim = 16
            self.map_noise = NoiseEmbedding(
                num_frequencies=args.hidden_dim * self.channel_mult_noise,
                hidden_dim=self.noise_level_dim,
                output_dim=self.noise_level_dim,
            )

        # Flow maps X_{s,t} take a second time: s goes through map_noise and
        # the step dt = t - s through map_dt, added to the time features
        # before the noise MLP. map_dt is zero-initialised, so a flow map
        # loaded from a flow matching (FM) checkpoint starts as the FM
        # velocity (one Euler step of the teacher).
        self.map_dt = None
        if args.model == "FlowMap":
            assert args.noise_embedding != "linear", (
                "FlowMap needs a time embedding (--noise_embedding "
                "positional or fourier)"
            )
            self.map_dt = DtEmbedding(
                num_channels=args.hidden_dim * self.channel_mult_noise,
                embedding_type=getattr(args, "fmap_dt_embedding", None)
                or "positional",
            )

        self.interior_embedder = make_mlp(
            [self.interior_input_dim] + self.grid_mlp_blueprint_end,
            cond_dim=self.noise_level_dim,
        )
        if self.boundary_forced:
            self.boundary_embedder = make_mlp(
                [self.boundary_dim] + self.grid_mlp_blueprint_end,
                cond_dim=self.noise_level_dim,
            )
        if self.atmosphere_forced and self.use_atmosphere_g2m:
            self.atmosphere_embedder = make_mlp(
                [self.atmosphere_dim] + self.grid_mlp_blueprint_end,
                cond_dim=self.noise_level_dim,
            )
        self.g2m_embedder = make_mlp(
            [g2m_dim] + self.edge_mlp_blueprint_end,
            cond_dim=self.noise_level_dim,
        )
        self.m2g_embedder = make_mlp(
            [m2g_dim] + self.edge_mlp_blueprint_end,
            cond_dim=self.noise_level_dim,
        )

        self.pre_mesh_proj = make_mlp(
            [hidden_dim_grid] + [args.hidden_dim],
            cond_dim=self.noise_level_dim,
        )
        self.post_mesh_proj = make_mlp(
            [args.hidden_dim] + [hidden_dim_grid],
            cond_dim=self.noise_level_dim,
        )

        self.g2m_gnn = FlexibleNet(
            edge_index=self.g2m_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_mesh_nodes,
            edge_dim=hidden_dim_edge,
            hidden_layers=args.hidden_layers,
            num_rec=self.num_grid_connected_mesh_nodes,
            cond_dim=self.noise_level_dim,
            propagation=True,
            aggr="mean",
        )
        self.encoding_grid_mlp = make_mlp(
            [hidden_dim_grid] + self.grid_mlp_blueprint_end,
            cond_dim=self.noise_level_dim,
        )

        # decoder
        self.m2g_gnn = FlexibleNet(
            edge_index=self.m2g_edge_index,
            send_node_dim=hidden_dim_grid,
            rec_node_dim=hidden_dim_grid,
            edge_dim=hidden_dim_edge,
            hidden_layers=args.hidden_layers,
            num_rec=self.num_grid_nodes,
            cond_dim=self.noise_level_dim,
            propagation=False,
            aggr="sum",
        )

        # Output mapping (hidden_dim_grid -> output_dim)
        # NOTE: cond_dim is still given, so that this shares the (conditional)
        # MLP parameter layout with the rest of the model, even though
        # layer_norm=False means no conditioning is actually applied here.
        self.output_map = make_mlp(
            [hidden_dim_grid] * (args.hidden_layers + 1)
            + [self.grid_output_dim],
            layer_norm=False,
            cond_dim=self.noise_level_dim,
        )  # No layer norm on this one

        # ---------------------------------------------------------------------
        # Mesh processor setup: hierarchical and single-level graphs use
        # different processing paths.
        if self.hierarchical:
            self.num_levels = len(self.mesh_static_features)

            # Number of mesh nodes at each level
            self.level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]  # Needs as list for later

            # Print some useful info
            utils.rank_zero_print("Loaded hierarchical graph with structure:")
            for level_index, level_mesh_size in enumerate(
                self.level_mesh_sizes
            ):
                same_level_edges = self.m2m_features[level_index].shape[0]
                utils.rank_zero_print(
                    f"level {level_index} - {level_mesh_size} nodes, "
                    f"{same_level_edges} same-level edges"
                )

                if level_index < (self.num_levels - 1):
                    up_edges = self.mesh_up_features[level_index].shape[0]
                    down_edges = self.mesh_down_features[level_index].shape[0]
                    utils.rank_zero_print(
                        f"  {level_index}<->{level_index + 1}"
                    )
                    utils.rank_zero_print(
                        f" - {up_edges} up edges, {down_edges} down edges"
                    )

            # Embedders
            # Assume all levels have same static feature dimensionality
            mesh_dim = self.mesh_static_features[0].shape[1]
            mesh_same_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            # Separate mesh node embedders for each level
            mesh_embedder_blueprint_0 = [hidden_dim_mesh_nodes] * (
                args.hidden_layers + 1
            )
            self.mesh_embedders = nn.ModuleList(
                [
                    make_mlp(
                        [mesh_dim] + mesh_embedder_blueprint_0,
                        cond_dim=self.noise_level_dim,
                    )
                ]
                + [
                    make_mlp(
                        [mesh_dim] + self.mlp_blueprint_end,
                        cond_dim=self.noise_level_dim,
                    )
                    for _ in range(self.num_levels - 1)
                ]
            )
            self.mesh_same_embedders = nn.ModuleList(
                [
                    make_mlp(
                        [mesh_same_dim] + self.mlp_blueprint_end,
                        cond_dim=self.noise_level_dim,
                    )
                    for _ in range(self.num_levels)
                ]
            )
            self.mesh_up_embedders = nn.ModuleList(
                [
                    make_mlp(
                        [mesh_up_dim] + self.mlp_blueprint_end,
                        cond_dim=self.noise_level_dim,
                    )
                    for _ in range(self.num_levels - 1)
                ]
            )
            self.mesh_down_embedders = nn.ModuleList(
                [
                    make_mlp(
                        [mesh_down_dim] + self.mlp_blueprint_end,
                        cond_dim=self.noise_level_dim,
                    )
                    for _ in range(self.num_levels - 1)
                ]
            )

            # Instantiate GNNs
            # Init GNNs
            self.mesh_init_gnns = nn.ModuleList(
                [
                    InteractionNet(
                        edge_index,
                        args.hidden_dim,
                        hidden_layers=args.hidden_layers,
                        cond_dim=self.noise_level_dim,
                    )
                    for edge_index in self.mesh_up_edge_index
                ]
            )

            # Read out GNNs
            self.mesh_read_gnns = nn.ModuleList(
                [
                    InteractionNet(
                        edge_index,
                        args.hidden_dim,
                        hidden_layers=args.hidden_layers,
                        update_edges=False,
                        cond_dim=self.noise_level_dim,
                    )
                    for edge_index in self.mesh_down_edge_index
                ]
            )

            # GraphFM-style vertical sweep processors
            self.mesh_down_gnns = nn.ModuleList(
                [
                    self.make_down_gnns(args)
                    for _ in range(args.processor_layers)
                ]
            )  # Nested lists (proc_steps, num_levels-1)
            self.mesh_down_same_gnns = nn.ModuleList(
                [
                    self.make_same_gnns(args)
                    for _ in range(args.processor_layers)
                ]
            )  # Nested lists (proc_steps, num_levels)
            self.mesh_up_gnns = nn.ModuleList(
                [self.make_up_gnns(args) for _ in range(args.processor_layers)]
            )  # Nested lists (proc_steps, num_levels-1)
            self.mesh_up_same_gnns = nn.ModuleList(
                [
                    self.make_same_gnns(args)
                    for _ in range(args.processor_layers)
                ]
            )  # Nested lists (proc_steps, num_levels)
        else:
            # Single-level mesh setup (GraphCast-like processor path)
            num_mesh = self.mesh_static_features.shape[0]
            num_m2m_edges = self.m2m_features.shape[0]
            utils.rank_zero_print(
                "Loaded non-hierarchical graph with structure:"
            )
            utils.rank_zero_print(
                f"  mesh nodes: {num_mesh}, same-level edges: {num_m2m_edges}"
            )

            mesh_dim = self.mesh_static_features.shape[1]
            m2m_dim = self.m2m_features.shape[1]
            mesh_embedder_blueprint = [hidden_dim_mesh_nodes] * (
                args.hidden_layers + 1
            )

            self.mesh_embedder = make_mlp(
                [mesh_dim] + mesh_embedder_blueprint,
                cond_dim=self.noise_level_dim,
            )
            self.m2m_embedder = make_mlp(
                [m2m_dim] + self.mlp_blueprint_end,
                cond_dim=self.noise_level_dim,
            )
            self.processor_gnns = nn.ModuleList(
                [
                    InteractionNet(
                        self.m2m_edge_index,
                        args.hidden_dim,
                        hidden_layers=args.hidden_layers,
                        aggr=args.mesh_aggr,
                        cond_dim=self.noise_level_dim,
                    )
                    for _ in range(args.processor_layers)
                ]
            )

        if getattr(args, "adaln_zero", False):
            utils.set_adaln_zero(self)

    # ----------------------------------------------------------------------------
    @property
    def num_grid_connected_mesh_nodes(self):
        """
        Get the total number of mesh nodes that have a connection to
        the grid (e.g. bottom level in a hierarchy)
        """
        if self.hierarchical:
            return self.mesh_static_features[0].shape[0]  # Bottom level
        return self.mesh_static_features.shape[0]

    def forward(
        self,
        x,
        noise_level,
        cond,
        boundary_forcing,
        atmosphere_forcing,
        dt=None,
    ):
        """
        Forward pass through the model
        x: (B, num_grid_nodes, feature_dim)
        noise_level: (B,) noise level / time, or (B, noise_dim) noise vector
            for the linear embedding
        cond: (B, num_grid_nodes, feature_dim)
        dt: (B,), flow map step t - s, only for models built with map_dt
        """
        # Mapping.
        if dt is None:
            emb = self.map_noise(noise_level)
        else:
            emb = self.map_noise(noise_level, extra=self.map_dt(dt))
        emb = emb.unsqueeze(1).expand(x.shape[0], 1, -1)
        prediction = self.predict_step(
            x, emb, cond, boundary_forcing, atmosphere_forcing
        )

        return prediction

    # BaseGraphModel methods
    def predict_step(
        self, x, emb, cond=None, boundary_forcing=None, atmosphere_forcing=None
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        x: (B, num_grid_nodes, feature_dim)
        emb: (B, num_grid_nodes, feature_dim)
        cond: (B, num_grid_nodes, feature_dim)
        """
        batch_size = x.shape[0]

        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        if cond is None:
            interior_input_list = [
                x,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ]
            if self.concat_atmosphere:
                interior_input_list.append(atmosphere_forcing)
            interior_features = torch.cat(interior_input_list, dim=-1)
        else:
            if self.remove_cond:
                # NOTE: We assume that prev_state is first in class_labels
                x = x - cond[:, :, : x.shape[-1]]
            interior_input_list = [
                x,
                cond,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ]
            if self.concat_atmosphere:
                interior_input_list.append(atmosphere_forcing)
            interior_features = torch.cat(interior_input_list, dim=-1)

        # Embed all features
        interior_emb = self.interior_embedder(
            interior_features, emb
        )  # (B, num_interior_nodes, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features, emb)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features, emb)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes(emb)
        grid_emb_list = [interior_emb]

        if self.boundary_forced:
            boundary_features = torch.cat(
                (
                    boundary_forcing,
                    self.expand_to_batch(
                        self.boundary_static_features, batch_size
                    ),
                ),
                dim=-1,
            )  # (B, num_boundary_nodes, interior_input_dim)
            boundary_emb = self.boundary_embedder(
                boundary_features, emb
            )  # (B, num_boundary_nodes, d_h)
            grid_emb_list.append(boundary_emb)

        if self.atmosphere_forced and self.use_atmosphere_g2m:
            atmosphere_features = torch.cat(
                (
                    atmosphere_forcing,
                    self.expand_to_batch(
                        self.atmosphere_static_features, batch_size
                    ),
                ),
                dim=-1,
            )  # (B, num_atmosphere_nodes, interior_input_dim)

            atmosphere_emb = self.atmosphere_embedder(
                atmosphere_features, emb
            )  # (B, num_atmosphere_nodes, d_h)
            grid_emb_list.append(atmosphere_emb)

        if len(grid_emb_list) == 1:
            # Only interior
            grid_emb = grid_emb_list[0]
        else:
            # NOTE: We here assume the order of grid node index is 1) interior,
            # 2) boundary, 3) atmosphere. This has to be followed also when
            # constructing g2m. Concatenates all existing embeddings.
            grid_emb = torch.cat(grid_emb_list, dim=1)
            # (B, num_grid_nodes, d_h)

        # Map from grid to mesh
        mesh_emb_expanded = self.expand_to_batch(
            mesh_emb, batch_size
        )  # (B, num_mesh_nodes, d_h)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded, emb
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep = interior_emb + self.encoding_grid_mlp(
            interior_emb, emb
        )  # (B, num_interior_nodes, d_h)

        # Project up mesh rep to hidden dim of graph
        mesh_rep = self.pre_mesh_proj(mesh_rep, emb)  # g -> m

        # Run processor step
        mesh_rep = self.process_step(mesh_rep, emb)  # m -> m

        # Project down mesh rep to hidden dim of grid
        mesh_rep = self.post_mesh_proj(mesh_rep, emb)  # m -> g

        # Map back from mesh to grid
        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded, emb
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep, emb
        )  # (B, num_grid_nodes, d_grid_out)

        return net_output

    # ----------------------------------------------------------------------------
    # BaseHiGraphModel methods
    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        if self.hierarchical:
            num_mesh_nodes = sum(
                node_feat.shape[0] for node_feat in self.mesh_static_features
            )
            num_mesh_nodes_ignore = (
                num_mesh_nodes - self.mesh_static_features[0].shape[0]
            )
        else:
            num_mesh_nodes = self.mesh_static_features.shape[0]
            num_mesh_nodes_ignore = 0
        return num_mesh_nodes, num_mesh_nodes_ignore

    def embedd_mesh_nodes(self, emb):
        """
        Embed static mesh features
        This embeds only bottom level, rest is done at beginning of
        processing step
        Returns tensor of shape (num_mesh_nodes[0], d_h)
        """
        if self.hierarchical:
            return self.mesh_embedders[0](self.mesh_static_features[0], emb)
        return self.mesh_embedder(self.mesh_static_features, emb)

    def process_step(self, mesh_rep, emb):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        if not self.hierarchical:
            batch_size = mesh_rep.shape[0]
            m2m_emb = self.m2m_embedder(self.m2m_features, emb)
            m2m_emb_expanded = self.expand_to_batch(m2m_emb, batch_size)

            for gnn in self.processor_gnns:
                mesh_rep, m2m_emb_expanded = gnn(
                    mesh_rep, mesh_rep, m2m_emb_expanded, emb
                )
            return mesh_rep

        batch_size = mesh_rep.shape[0]

        # EMBED REMAINING MESH NODES (levels >= 1) -
        # Create list of mesh node representations for each level,
        # each of size (B, num_mesh_nodes[l], d_h)
        mesh_rep_levels = [mesh_rep] + [
            self.expand_to_batch(
                embedder(node_static_features, emb), batch_size
            )
            for embedder, node_static_features in zip(
                list(self.mesh_embedders)[1:],
                list(self.mesh_static_features)[1:],
            )
        ]

        # - EMBED EDGES -
        # Embed edges, expand with batch dimension
        mesh_same_rep = [
            self.expand_to_batch(embedder(edge_feat, emb), batch_size)
            for embedder, edge_feat in zip(
                self.mesh_same_embedders, self.m2m_features
            )
        ]
        mesh_up_rep = [
            self.expand_to_batch(embedder(edge_feat, emb), batch_size)
            for embedder, edge_feat in zip(
                self.mesh_up_embedders, self.mesh_up_features
            )
        ]
        mesh_down_rep = [
            self.expand_to_batch(embedder(edge_feat, emb), batch_size)
            for embedder, edge_feat in zip(
                self.mesh_down_embedders, self.mesh_down_features
            )
        ]

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, num_mesh_nodes[l-1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(
                send_node_rep, rec_node_rep, edge_rep, emb
            )

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = (
                new_node_rep  # (B, num_mesh_nodes[l], d_h)
            )
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep, emb
        )

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
            range(self.num_levels - 2, -1, -1), reversed(self.mesh_read_gnns)
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, num_mesh_nodes[l+1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(send_node_rep, rec_node_rep, edge_rep, emb)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = (
                new_node_rep  # (B, num_mesh_nodes[l], d_h)
            )

        # Return only bottom level representation
        return mesh_rep_levels[0]  # (B, num_mesh_nodes[0], d_h)

    # ----------------------------------------------------------------------------
    # GraphFM specific methods
    def make_same_gnns(self, args):
        """
        Make intra-level GNNs.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    cond_dim=self.noise_level_dim,
                )
                for edge_index in self.m2m_edge_index
            ]
        )

    def make_up_gnns(self, args):
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        return nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    cond_dim=self.noise_level_dim,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )

    def make_down_gnns(self, args):
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        return nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    cond_dim=self.noise_level_dim,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )

    def mesh_down_step(
        self,
        mesh_rep_levels,
        mesh_same_rep,
        mesh_down_rep,
        down_gnns,
        same_gnns,
        emb,
    ):
        """
        Run down-part of vertical processing, sequentially alternating between
        processing using down edges and same-level edges.
        """
        # Run same level processing on level L
        mesh_rep_levels[-1], mesh_same_rep[-1] = same_gnns[-1](
            mesh_rep_levels[-1], mesh_rep_levels[-1], mesh_same_rep[-1], emb
        )

        # Let level_l go from L-1 to 0
        for level_l, down_gnn, same_gnn in zip(
            range(self.num_levels - 2, -1, -1),
            reversed(down_gnns),
            reversed(same_gnns[:-1]),
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, N_mesh[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            down_edge_rep = mesh_down_rep[level_l]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply down GNN
            new_node_rep, mesh_down_rep[level_l] = down_gnn(
                send_node_rep, rec_node_rep, down_edge_rep, emb
            )

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep, emb
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(
        self,
        mesh_rep_levels,
        mesh_same_rep,
        mesh_up_rep,
        up_gnns,
        same_gnns,
        emb,
    ):
        """
        Run up-part of vertical processing, sequentially alternating between
        processing using up edges and same-level edges.
        """

        # Run same level processing on level 0
        mesh_rep_levels[0], mesh_same_rep[0] = same_gnns[0](
            mesh_rep_levels[0], mesh_rep_levels[0], mesh_same_rep[0], emb
        )

        # Let level_l go from 1 to L
        for level_l, (up_gnn, same_gnn) in enumerate(
            zip(up_gnns, same_gnns[1:]), start=1
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, N_mesh[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            up_edge_rep = mesh_up_rep[level_l - 1]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply up GNN
            new_node_rep, mesh_up_rep[level_l - 1] = up_gnn(
                send_node_rep, rec_node_rep, up_edge_rep, emb
            )
            # (B, N_mesh[l], d_h) and (B, M_up[l-1], d_h)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep, emb
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep, emb
    ):
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        for down_gnns, down_same_gnns, up_gnns, up_same_gnns in zip(
            self.mesh_down_gnns,
            self.mesh_down_same_gnns,
            self.mesh_up_gnns,
            self.mesh_up_same_gnns,
        ):
            # Down
            mesh_rep_levels, mesh_same_rep, mesh_down_rep = self.mesh_down_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_down_rep,
                down_gnns,
                down_same_gnns,
                emb,
            )

            # Up
            mesh_rep_levels, mesh_same_rep, mesh_up_rep = self.mesh_up_step(
                mesh_rep_levels,
                mesh_same_rep,
                mesh_up_rep,
                up_gnns,
                up_same_gnns,
                emb,
            )

        # Note: We return all, even though only down edges really are used later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.
class LinearNoiseEmbedding(nn.Module):
    def __init__(self, noise_dim=32, hidden_dim=128, output_dim=128):
        super(LinearNoiseEmbedding, self).__init__()
        self.linear = nn.Linear(noise_dim, noise_dim)
        self.mlp = NoiseLevelMLP(
            input_dim=noise_dim, output_dim=output_dim, hidden_dim=hidden_dim
        )

    def forward(self, noise):
        noise_level_encoding = self.linear(noise)  # (batch_size, noise_dim)
        noise_level_encoding = self.mlp(noise_level_encoding)
        if noise_level_encoding.dim() == 1:
            noise_level_encoding = noise_level_encoding.unsqueeze(
                0
            )  # Add back batch dimension if missing
        return noise_level_encoding


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal embedding with log-spaced frequencies <= 1 (EDM / DDPM++).
    Smooth in its input, unlike FourierEmbedding with scale 16, which matters
    when the network is differentiated in time (flow maps).
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            dtype=torch.float32,
            device=x.device,
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class DtEmbedding(nn.Module):
    """
    Embedding of the flow map step dt = t - s, (B,) -> (B, num_channels),
    added to the time features before the noise MLP. The output projection
    is zero-initialised, so the embedding is zero at initialisation.
    """

    def __init__(self, num_channels, embedding_type="positional"):
        super().__init__()
        if embedding_type == "positional":
            self.embed = PositionalEmbedding(num_channels, endpoint=True)
        else:
            self.embed = FourierEmbedding(num_channels)
        self.proj = nn.Linear(num_channels, num_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, dt):
        return self.proj(self.embed(dt))


class NoiseLevelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=16):
        super(NoiseLevelMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, emb):
        x = silu(self.fc1(emb))
        x = silu(self.fc2(x))
        x = self.fc3(x)
        return x  # Output noise-level encoding (batch_size, output_dim)


class NoiseEmbedding(nn.Module):
    def __init__(
        self, num_frequencies=32, base_period=16, output_dim=16, hidden_dim=128
    ):
        super(NoiseEmbedding, self).__init__()
        self.fourier_transform = FourierEmbedding(
            num_channels=num_frequencies, scale=base_period
        )
        self.mlp = NoiseLevelMLP(
            input_dim=num_frequencies,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, log_noise_levels, extra=None):
        """
        log_noise_levels: (B,)
        extra: (B, num_frequencies), optional features added before the MLP
        """
        fourier_features = self.fourier_transform(log_noise_levels)
        fourier_features = (
            fourier_features.reshape(fourier_features.shape[0], 2, -1)
            .flip(1)
            .reshape(*fourier_features.shape)
            .contiguous()
        )  # swap sin/cos
        if extra is not None:
            fourier_features = fourier_features + extra
        noise_level_encoding = self.mlp(fourier_features)
        return noise_level_encoding


class PositionalNoiseEmbedding(nn.Module):
    def __init__(self, num_frequencies=32, output_dim=16, hidden_dim=128):
        super().__init__()
        self.positional_transform = PositionalEmbedding(
            num_channels=num_frequencies
        )
        self.mlp = NoiseLevelMLP(
            input_dim=num_frequencies,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, noise_levels, extra=None):
        """
        noise_levels: (B,)
        extra: (B, num_frequencies), optional features added before the MLP
        """
        features = self.positional_transform(noise_levels)
        if extra is not None:
            features = features + extra
        return self.mlp(features)
