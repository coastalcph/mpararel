#!/bin/bash
#SBATCH --job-name=translate-templates
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--output_folder ${WORKDIR}/data/multilingual/pararel_bing \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator bing