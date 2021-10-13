#!/bin/bash
#SBATCH --job-name=bing-translation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00

export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

mkdir -p ${WORKDIR}/data/multilingual_logging/pararel_bing
python dataset/translate_templates.py translate_folder \
    --templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
    --output_folder ${WORKDIR}/data/multilingual_logging/pararel_bing \
    --language_mapping_file ${WORKDIR}/dataset/languages_mapping_2.txt \
    --only_wiki_codes zh zh-classical \
    --translator bing