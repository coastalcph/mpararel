source mpararel-venv/bin/activate
export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

# (1) Download TREx.
wget https://dl.fbaipublicfiles.com/LAMA/data.zip -P ${WORKDIR}/data/lama
unzip ${WORKDIR}/data/lama/data.zip -d ${WORKDIR}/data/lama && rm ${WORKDIR}/data/lama/data.zip
mv ${WORKDIR}/data/lama/data/TREx ${WORKDIR}/data/lama/TREx && rm -R ${WORKDIR}/data/lama/data

# (2) Download entity data.
mkdir -p ${WORKDIR}/data/wikidata_entities
python dataset/download_trexentities.py \
--datapath ${WORKDIR}/data/lama/TREx \
--outpath ${WORKDIR}/data/wikidata_entities

# (3) Translate TREx entities.
mkdir -p ${WORKDIR}/data/translations/t_rex_translation_zh_corrected
python dataset/translate_trex.py \
--data ${WORKDIR}/data/lama/TREx \
--entities ${WORKDIR}/data/wikidata_entities \
--outpath ${WORKDIR}/data/translations/t_rex_translation_zh_corrected \
--languagemapping ${WORKDIR}/dataset/languages_mapping.txt

# (4) Get ParaRel data.
mkdir ${WORKDIR}/data/pararel
git clone git@github.com:yanaiela/pararel.git
mv pararel/data/* ${WORKDIR}/data/pararel
rm -r pararel

# (5) Translate ParaRel
# E.g. Translate ParaRel with Google.
mkdir -p ${WORKDIR}/data/translations/pararel_google
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
--output_folder ${WORKDIR}/data/translations/pararel_google \
--language_mapping_file mbertlangs.txt \
--translator google

# E.g. Translate ParaRel on the populated templates with Bing
mkdir -p ${WORKDIR}/data/translations/pararel_populated_bing
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
--translate_populated_templates \
--tuples_folder ${WORKDIR}/data/translations/t_rex_translation \
--output_folder ${WORKDIR}/data/translations/pararel_populated_bing \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator bing

# Note that the m2m100_big, mbart50_en2m models need GPU and opus_mt is also
# faster with a GPU. Check https://github.com/UKPLab/EasyNMT#available-models

# (6) Fix templates translations errors in place.
cp -r ${WORKDIR}/data/translations/pararel_google ${WORKDIR}/data/translations/pararel_google_fixed
cp -r ${WORKDIR}/data/translations/pararel_bing ${WORKDIR}/data/translations/pararel_bing_fixed
cp -r ${WORKDIR}/data/translations/pararel_m2m100_big ${WORKDIR}/data/translations/pararel_m2m100_big_fixed
cp -r ${WORKDIR}/data/translations/pararel_mbart50_en2m ${WORKDIR}/data/translations/pararel_mbart50_en2m_fixed
cp -r ${WORKDIR}/data/translations/pararel_opus_mt ${WORKDIR}/data/translations/pararel_opus_mt_fixed
python dataset/translate_templates.py fix_translated_dirs \
	--templates_folder_glob=${WORKDIR}/data/translations/pararel_*_fixed

# (7) Copy only clean and valid templates and relations to an output folder.
python dataset/cleanup.py \
    --tuples_folder ${WORKDIR}/data/translations/t_rex_translation_zh_corrected/ \
	--templates_folders_glob=${WORKDIR}/data/translations/pararel*fixed \
	--correct_chinese_wiki_code \
    --out_folder ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/

# (8) Create the mParaRel resource by selecting the agreed template translations
# and the languages with a minimum coverage.
python dataset/create_mpararel.py \
	--translations_folders_glob=${WORKDIR}/data/cleaned_mtrex_and_mpatterns/patterns/* \
	--tuples_folder ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/tuples \
    --pararel_patterns_folder=$WORKDIR/data/pararel/pattern_data/graphs_json \
    --mlama_folder=$WORKDIR/data/mlama1.1 \
	--min_templates_per_relation 0.0 \
	--min_phrases_per_relation 0.0 \
	--min_relations_count 0.6 \
	--min_total_phrases 0.2 \
	--out_folder ${WORKDIR}/data/mpararel_with_mlama_zh_corrected/patterns \
	--wandb_run_name mpararel_with_mlama_zh_corrected
mv ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/tuples ${WORKDIR}/data/mpararel_with_mlama_zh_corrected/
rm -r ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/