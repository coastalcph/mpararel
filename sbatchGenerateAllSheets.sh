#!/bin/bash
#SBATCH --job-name=review-sheets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-05:00:00

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

python dataset/crowdsourcing/generate_all_sheets.py \
    --mpararel_folder=$WORKDIR/data/mpararel_no_populated \
    --pararel_patterns_folder=$WORKDIR/data/pararel/pattern_data/graphs_json \
    --language_mapping_file=$WORKDIR/dataset/languages_mapping_2.txt