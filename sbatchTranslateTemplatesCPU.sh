#!/bin/bash
#SBATCH --job-name=google-populated-translation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--translate_populated_templates --tuples_folder ${WORKDIR}/data/multilingual \
--output_folder ${WORKDIR}/data/multilingual/pararel_populated_google \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator google