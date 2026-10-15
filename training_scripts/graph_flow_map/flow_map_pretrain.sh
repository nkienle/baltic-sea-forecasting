# FlowMap part 1/3: 1 deg, ar=1, flow-map loss (lsd + diagonal flow matching).
#   sbatch training_scripts/train.sh training_scripts/graph_flow_map/flow_map_pretrain.sh
# Extra arguments override the ones below, e.g. to resume/extend this part:
#   ... flow_map_pretrain.sh --load saved_models/<run>/last.ckpt
# NOTE: fp32, torch.func.jvp (lsd/esd losses) is not tested under bf16.
srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_density.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model FlowMap \
    --scheduler flow_map_pretrain \
    --epochs 175 \
    --graph global_cluster_1_deg_15_refinement_4_levels \
    --hidden_dim 1024 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 1 \
    --noise_embedding positional \
    --pred_residual \
    --lr 0.0005 \
    --min_lr 0.00001 \
    --grad_clip 1.0 \
    --fmap_pair_mode alternate \
    --fmap_loss psd \
    --fmap_diag_fraction 0.75 \
    --fmap_ema \
    --fmap_ema_decay 0.9999 \
    --fmap_val_sample_steps_extra 20 \
    --ensemble_size 5 \
    --n_example_pred 1 \
    --val_interval 5 \
    --ar_steps_eval 1 \
    --val_steps_to_log 1 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --optimizer "muon_flat" \
    "$@"
