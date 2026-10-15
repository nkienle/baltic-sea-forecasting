# FlowMap part 2/3: 0.25 deg, ar=1, flow-map loss. Starts from the last
# checkpoint of part 1:
#   sbatch training_scripts/train.sh training_scripts/graph_flow_map/flow_map_finetune.sh \
#       --load saved_models/<part-1 run>/last.ckpt
# To train this part for longer (until the 1-jump samples in the validation
# plots are good), re-run it from its own checkpoint with a larger --epochs
# (the total length of the part); optimizer state is then kept:
#   ... flow_map_finetune.sh --load saved_models/<part-2 run>/last.ckpt --epochs 150
srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_4_density.yaml \
    --num_workers 2 \
    --precision 32 \
    --model FlowMap \
    --scheduler flow_map_finetune \
    --epochs 100 \
    --graph global_cluster_1_4_deg_20_refinement_4_levels \
    --hidden_dim 256 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 3 \
    --noise_embedding positional \
    --pred_residual \
    --lr 0.0001 \
    --min_lr 0.00001 \
    --grad_clip 1.0 \
    --fmap_loss lsd \
    --fmap_diag_fraction 0.75 \
    --fmap_ema \
    --fmap_ema_decay 0.9999 \
    --fmap_val_sample_steps_extra 4 \
    --ensemble_size 5 \
    --n_example_pred 1 \
    --val_interval 5 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 4 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    "$@"
