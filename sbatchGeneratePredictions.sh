#!/bin/bash
#SBATCH --job-name=pred_mpararel
#SBATCH -p gpu --gres=gpu:titanrtx:1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

python evaluate_consistency/get_model_predictions.py \
    --mpararel_folder=$WORKDIR/data/mpararel_00_00_06_02_logging \
    --model_name="bert-base-multilingual-cased" --batch_size=32 \
    --output_folder=$WORKDIR/data/mpararel_predictions/mbert_cased