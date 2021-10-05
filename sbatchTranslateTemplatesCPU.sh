#!/bin/bash
#SBATCH --job-name=google-translation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

mkdir -p ${WORKDIR}/data/multilingual_logging/pararel_google
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
--output_folder ${WORKDIR}/data/multilingual_logging/pararel_google \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator google