#!/bin/bash
#SBATCH --job-name=opus-translate
#SBATCH -p gpu --gres=gpu:titanrtx:1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

mkdir -p ${WORKDIR}/data/multilingual/pararel_opus_mt
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--output_folder ${WORKDIR}/data/multilingual/pararel_opus_mt \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator opus_mt