source mpararel-venv/bin/activate
export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

# (1) Download TREx.
wget https://dl.fbaipublicfiles.com/LAMA/data.zip -P ${WORKDIR}
unzip ${WORKDIR}/data.zip -d ${WORKDIR} && rm ${WORKDIR}/data.zip

# (2) Download entity data
mkdir -p ${WORKDIR}/data/wikidata_entities
python download_trexentities.py \
--datapath ${WORKDIR}/data/TREx \
--outpath ${WORKDIR}/data/wikidata_entities

# (3) Translate TREx entities.
mkdir -p ${WORKDIR}/data/multilingual/t_rex_translation
python translate_trex.py \
--data ${WORKDIR}/data/TREx \
--entities ${WORKDIR}/data/wikidata_entities \
--outpath ${WORKDIR}/data/multilingual/t_rex_translation \
--languagemapping mbertlangs.txt

# (4) Get ParaRel data.
mkdir ${WORKDIR}/data/pararel
git clone git@github.com:yanaiela/pararel.git
mv pararel/data/pattern_data/graphs_json/* ${WORKDIR}/data/pararel
rm -r pararel

# (5) Translate ParaRel
# E.g. Translate ParaRel with Google.
mkdir -p ${WORKDIR}/data/multilingual/pararel_google
python translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--output_folder ${WORKDIR}/data/multilingual/pararel_google \
--language_mapping_file mbertlangs.txt \
--translator google

# E.g. Translate ParaRel on the populated templates with Bing
mkdir -p ${WORKDIR}/data/multilingual/pararel_populated_bing
python dataset/translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--translate_populated_templates \
--tuples_folder ${WORKDIR}/data/multilingual/t_rex_translation \
--output_folder ${WORKDIR}/data/multilingual/pararel_populated_bing \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator bing

# Note that the m2m100_big, mbart50_en2m models need GPU and opus_mt is also
# faster with a GPU. Check https://github.com/UKPLab/EasyNMT#available-models

# (6) Fix templates translations errors in place.
cp -r ${WORKDIR}/data/multilingual/pararel_google ${WORKDIR}/data/multilingual/pararel_google_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_bing ${WORKDIR}/data/multilingual/pararel_bing_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_m2m100_big ${WORKDIR}/data/multilingual/pararel_m2m100_big_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_mbart50_en2m ${WORKDIR}/data/multilingual/pararel_mbart50_en2m_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_opus_mt ${WORKDIR}/data/multilingual/pararel_opus_mt_fixed

cp -r ${WORKDIR}/data/multilingual/pararel_populated_google_with_metadata ${WORKDIR}/data/multilingual/pararel_populated_google_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_populated_bing_with_metadata ${WORKDIR}/data/multilingual/pararel_populated_bing_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_populated_m2m100_big_with_metadata ${WORKDIR}/data/multilingual/pararel_populated_m2m100_big_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_populated_mbart50_en2m_with_metadata ${WORKDIR}/data/multilingual/pararel_populated_mbart50_en2m_fixed
cp -r ${WORKDIR}/data/multilingual/pararel_populated_opus_mt_with_metadata ${WORKDIR}/data/multilingual/pararel_populated_opus_mt_fixed
python translate_templates.py fix_translated_dirs \
	--templates_folder_glob ${WORKDIR}/data/multilingual/pararel_*_fixed

# (7) Copy only clean and valid templates and relations to an output folder.
mkdir -p ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/
python cleanup.py \
    --tuples_folder ${WORKDIR}/data/multilingual/t_rex_translation \
	--templates_folders_glob=${WORKDIR}/data/multilingual/pararel*fixed \
    --out_folder ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/
rm -r ${WORKDIR}/data/multilingual/pararel_*_fixed

# (8) Create the mParaRel resource by selecting the agreed template translations
# and the languages with a minimum coverage. 
# TODO: add to the count the wikidata entities
mkdir -p ${WORKDIR}/mpararel
python create_mpararel.py \
	--translations_folders_glob=${WORKDIR}/data/cleaned_mtrex_and_mpatterns/patterns/* \
	--min_templates_per_relation 0.2 \
	--min_relations_count 1.0 \
	--out_folder ${WORKDIR}/mpararel/patterns
mv ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/tuples ${WORKDIR}/mpararel/tuples
rm -r ${WORKDIR}/data/cleaned_mtrex_and_mpatterns/