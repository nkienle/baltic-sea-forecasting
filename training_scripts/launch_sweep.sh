#!/bin/bash
# Queue one Slurm job per run configuration below.
#
#   bash training_scripts/launch_sweep.sh              # submit everything
#   DRY_RUN=1 bash training_scripts/launch_sweep.sh    # only print the sbatch commands
#   DEPEND=12345 bash training_scripts/launch_sweep.sh # start after job 12345 finishes
#
# The training script to sweep can be overridden:
#   bash training_scripts/launch_sweep.sh training_scripts/seacast/seacast_finetune.sh

cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting || exit 1

TRAIN_SCRIPT="${1:-training_scripts/seacast/seacast_pretrain.sh}"

# One line per run: the flags to append to the training command. They override
# the defaults in the training script, so anything can be varied here.
RUNS=(
    # "--hidden_dim 256 --graph global_cluster_1_deg_20_refinement_4_levels"
    "--hidden_dim 512 --graph global_cluster_1_deg_20_refinement_4_levels --batch_size 1"
    "--hidden_dim 1024 --graph global_cluster_1_deg_20_refinement_4_levels --batch_size 1"

    # "--hidden_dim 256 --graph global_cluster_1_deg_20_refinement_4_levels"
    "--hidden_dim 256 --graph global_cluster_1_deg_15_refinement_4_levels --batch_size 1"
    "--hidden_dim 256 --graph global_cluster_1_deg_10_refinement_4_levels --batch_size 1"

    "--hidden_dim 1024 --graph global_cluster_1_deg_15_refinement_4_levels --batch_size 1"
    "--hidden_dim 672 --graph global_cluster_1_deg_10_refinement_4_levels --batch_size 1"
)

# Build a short job name from the run flags. This ends up in the log file name
# via the %x in the --output of train.sh, so keep it short and plain.
make_tag() {
    local tokens=($1) hidden_dim graph stoch="" i j
    for ((i = 0; i < ${#tokens[@]}; i++)); do
        case "${tokens[i]}" in
            --hidden_dim) hidden_dim="${tokens[i + 1]}" ;;
            --graph) graph="${tokens[i + 1]}" ;;
            --stochasticity)
                # graph_crps runs: name them by their sources, e.g. crps_fgn_mc
                stoch="crps"
                for ((j = i + 1; j < ${#tokens[@]}; j++)); do
                    [[ "${tokens[j]}" == --* ]] && break
                    stoch="${stoch}_${tokens[j]}"
                done
                ;;
        esac
    done
    if [ -n "$stoch" ] && [ -z "$hidden_dim$graph" ]; then
        echo "$stoch"
        return
    fi
    graph="${graph#global_cluster_}"
    graph="${graph%_refinement_*}"
    echo "h${hidden_dim}_${graph}${stoch:+_$stoch}"
}

for run in "${RUNS[@]}"; do
    sbatch_args=(-J "SeaCast_$(make_tag "$run")")
    [ -n "$DEPEND" ] && sbatch_args+=(--dependency="afterany:$DEPEND")

    # $run is deliberately unquoted so it splits into separate arguments
    if [ -n "$DRY_RUN" ]; then
        echo "sbatch ${sbatch_args[*]} training_scripts/train.sh $TRAIN_SCRIPT $run"
    else
        job_id=$(sbatch --parsable "${sbatch_args[@]}" \
            training_scripts/train.sh "$TRAIN_SCRIPT" $run)
        echo "Submitted $job_id: $run"
    fi
done
