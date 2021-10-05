#!/bin/bash
#SBATCH --job-name=opus_translation
#SBATCH -p gpu --gres=gpu:titanrtx:1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

mkdir -p ${WORKDIR}/data/multilingual_logging/pararel_opus_mt
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
--output_folder ${WORKDIR}/data/multilingual_logging/pararel_opus_mt \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator opus_mt