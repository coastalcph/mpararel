#!/bin/bash
#SBATCH --job-name=bart-translate
#SBATCH -p gpu --gres=gpu:titanrtx:1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

mkdir -p ${WORKDIR}/data/multilingual/mbart50_en2m
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--output_folder ${WORKDIR}/data/multilingual/mbart50_en2m \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator mbart50_en2m