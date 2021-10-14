source mpararel-venv/bin/activate
export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

# (1) Download TREx.
wget https://dl.fbaipublicfiles.com/LAMA/data.zip -P ${WORKDIR}
unzip ${WORKDIR}/data.zip -d ${WORKDIR} && rm ${WORKDIR}/data.zip

# (2) Download entity data
mkdir -p ${WORKDIR}/data/wikidata_entities
python dataset/download_trexentities.py \
--datapath ${WORKDIR}/data/TREx \
--outpath ${WORKDIR}/data/wikidata_entities

# (3) Translate TREx entities.
mkdir -p ${WORKDIR}/data/multilingual/t_rex_translation
python dataset/translate_trex.py \
--data ${WORKDIR}/data/TREx \
--entities ${WORKDIR}/data/wikidata_entities \
--outpath ${WORKDIR}/data/multilingual/t_rex_translation \
--languagemapping mbertlangs.txt

# (4) Get ParaRel data.
mkdir ${WORKDIR}/data/pararel
git clone git@github.com:yanaiela/pararel.git
mv pararel/data/* ${WORKDIR}/data/pararel
rm -r pararel

# (5) Translate ParaRel
# E.g. Translate ParaRel with Google.
mkdir -p ${WORKDIR}/data/multilingual/pararel_google
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
--output_folder ${WORKDIR}/data/multilingual/pararel_google \
--language_mapping_file mbertlangs.txt \
--translator google

# E.g. Translate ParaRel on the populated templates with Bing
mkdir -p ${WORKDIR}/data/multilingual/pararel_populated_bing
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel/pattern_data/graphs_json/ \
--translate_populated_templates \
--tuples_folder ${WORKDIR}/data/multilingual/t_rex_translation \
--output_folder ${WORKDIR}/data/multilingual/pararel_populated_bing \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator bing

# Note that the m2m100_big, mbart50_en2m models need GPU and opus_mt is also
# faster with a GPU. Check https://github.com/UKPLab/EasyNMT#available-models

# (6) Fix templates translations errors in place.
cp -r ${WORKDIR}/data/multilingual_logging/pararel_google ${WORKDIR}/data/multilingual_logging/pararel_google_fixed
cp -r ${WORKDIR}/data/multilingual_logging/pararel_bing ${WORKDIR}/data/multilingual_logging/pararel_bing_fixed
cp -r ${WORKDIR}/data/multilingual_logging/pararel_m2m100_big ${WORKDIR}/data/multilingual_logging/pararel_m2m100_big_fixed
cp -r ${WORKDIR}/data/multilingual_logging/pararel_mbart50_en2m ${WORKDIR}/data/multilingual_logging/pararel_mbart50_en2m_fixed
cp -r ${WORKDIR}/data/multilingual_logging/pararel_opus_mt ${WORKDIR}/data/multilingual_logging/pararel_opus_mt_fixed
python dataset/translate_templates.py fix_translated_dirs \
	--templates_folder_glob=${WORKDIR}/data/multilingual_logging/pararel_*_fixed

# (7) Copy only clean and valid templates and relations to an output folder.
python dataset/cleanup.py \
    --tuples_folder ${WORKDIR}/data/multilingual/t_rex_translation \
	--templates_folders_glob=${WORKDIR}/data/multilingual_logging/pararel*fixed \
    --out_folder ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/

# (8) Create the mParaRel resource by selecting the agreed template translations
# and the languages with a minimum coverage.
python dataset/create_mpararel.py \
	--translations_folders_glob=${WORKDIR}/data/cleaned_mtrex_and_mpatterns/patterns/* \
	--tuples_folder ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/tuples \
	--min_templates_per_relation 0.0 \
	--min_phrases_per_relation 0.0 \
	--min_relations_count 0.6 \
	--min_total_phrases 0.2 \
	--out_folder ${WORKDIR}/data/mpararel_no_populated_with_chinese/patterns
	--wandb_run_name mpararel_no_populated_with_chinese
mv ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/tuples ${WORKDIR}/data/mpararel_no_populated_with_chinese/
rm -r ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/