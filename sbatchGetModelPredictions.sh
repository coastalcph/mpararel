#!/bin/bash
#SBATCH --job-name=pred_mpararel
#SBATCH --ntasks=1 --cpus-per-task=10
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=3-00:00:00

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

python evaluate_consistency/get_model_predictions.py \
    --mpararel_folder=$WORKDIR/data/mpararel_with_mlama \
    --model_name="bert-base-multilingual-cased" --batch_size=32 \
    --output_folder=$WORKDIR/data/mpararel_predictions/mbert_cased_with_mlama \
    --path_to_existing_predictions=$WORKDIR/data/mpararel_predictions/mbert_cased \
    --cpus 10