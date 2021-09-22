#!/bin/bash
#SBATCH --job-name=bing-populated-translation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

mkdir -p ${WORKDIR}/data/multilingual/pararel_populated_bing
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
--translate_populated_templates \
--tuples_folder ${WORKDIR}/data/multilingual/t_rex_translation \
--output_folder ${WORKDIR}/data/multilingual/pararel_populated_bing \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator bing