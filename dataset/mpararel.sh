export PYTHONPATH=$(pwd)
export WORKDIR="/home/wsr217/mpararel"

# Download TREx.
wget https://dl.fbaipublicfiles.com/LAMA/data.zip -P ${WORKDIR}
unzip ${WORKDIR}/data.zip -d ${WORKDIR} && rm ${WORKDIR}/data.zip

# Download entity data
mkdir -p ${WORKDIR}/data/wikidata_entities
python download_trexentities.py \
--datapath ${WORKDIR}/data/TREx \
--outpath ${WORKDIR}/data/wikidata_entities

# Translate TREx entities.
mkdir -p ${WORKDIR}/data/multilingual/t_rex_translation
python translate_trex.py \
--data ${WORKDIR}/data/TREx \
--entities ${WORKDIR}/data/wikidata_entities \
--outpath ${WORKDIR}/data/multilingual/t_rex_translation \
--languagemapping mbertlangs.txt

# Get ParaRel data.
mkdir ${WORKDIR}/data/pararel
git clone git@github.com:yanaiela/pararel.git
mv pararel/data/pattern_data/graphs_json/* ${WORKDIR}/data/pararel
rm -r pararel

# Translate ParaRel with Google.
mkdir -p ${WORKDIR}/data/multilingual/pararel_google
python translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--output_folder ${WORKDIR}/data/multilingual/pararel_google \
--language_mapping_file mbertlangs.txt \
--translator google

# Translate ParaRel with Bing.
mkdir -p ${WORKDIR}/data/multilingual/pararel_bing
python translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--output_folder ${WORKDIR}/data/multilingual/pararel_bing \
--language_mapping_file languages_mapping.txt \
--translator bing

# Translate ParaRel by populating the templates.
mkdir -p ${WORKDIR}/data/multilingual/pararel_populated_google
python translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--translate_populated_templates --tuples_folder ${WORKDIR}/data/multilingual \
--output_folder ${WORKDIR}/data/multilingual/pararel_populated_google \
--language_mapping_file ${WORKDIR}/dataset/languages_mapping.txt \
--translator google

# Fix templates in place.
cp -r ${WORKDIR}/data/multilingual/pararel_google ${WORKDIR}/data/multilingual/pararel_google_fixed
python translate_templates.py fix_translated_dirs \
	--templates_folder ${WORKDIR}/data/multilingual/pararel_google_fixed

cp -r ${WORKDIR}/data/multilingual/pararel_bing ${WORKDIR}/data/multilingual/pararel_bing_fixed
python translate_templates.py fix_translated_dirs \
	--templates_folder ${WORKDIR}/data/multilingual/pararel_bing_fixed

cp -r ${WORKDIR}/data/multilingual/pararel_m2m100_big ${WORKDIR}/data/multilingual/pararel_m2m100_big_fixed
python translate_templates.py fix_translated_dirs \
	--templates_folder ${WORKDIR}/data/multilingual/pararel_m2m100_big_fixed

cp -r ${WORKDIR}/data/multilingual/pararel_mbart50_en2m ${WORKDIR}/data/multilingual/pararel_mbart50_en2m_fixed
python translate_templates.py fix_translated_dirs \
	--templates_folder ${WORKDIR}/data/multilingual/pararel_mbart50_en2m_fixed

cp -r ${WORKDIR}/data/multilingual/pararel_opus_mt ${WORKDIR}/data/multilingual/pararel_opus_mt_fixed
python translate_templates.py fix_translated_dirs \
	--templates_folder ${WORKDIR}/data/multilingual/pararel_opus_mt_fixed

# Copy valid templates and relations to an output folder
mkdir -p ${WORKDIR}/cleaned_datasets/
python cleanup.py \
    --tuples_folder ${WORKDIR}/data/multilingual/t_rex_translation \
	--templates_folders_glob=${WORKDIR}/data/multilingual/pararel*fixed \
    --out_folder ${WORKDIR}/cleaned_datasets
