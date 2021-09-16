#!/bin/bash
#SBATCH --job-name=bart-translate
#SBATCH -p gpu --gres=gpu:titanrtx:1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

mkdir -p ${WORKDIR}/data/multilingual/pararel_populated_opus_mt
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--translate_populated_templates \
--tuples_folder ${WORKDIR}/data/multilingual/t_rex_translation \
--output_folder ${WORKDIR}/data/multilingual/pararel_populated_opus_mt \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator opus_mt