#!/bin/bash
#SBATCH --job-name=consistency
#SBATCH -p gpu --gres=gpu:titanrtx:1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

pararel/consistency/encode_consistency_probe.py \
        --data_file ${WORKDIR}/data/pararel/trex_lms_vocab/P449.jsonl \
        --lm 'bert-base-cased' \
        --graph ${WORKDIR}/data/pararel/pattern_data/graphs/P449.graph \
        --gpu 0 \
        --wandb \
        --use_targets