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
mkdir -p ${WORKDIR}/data/multilingual
python translate_trex.py \
--data ${WORKDIR}/data/TREx \
--entities ${WORKDIR}/data/wikidata_entities \
--outpath ${WORKDIR}/data/multilingual \
--languagemapping mbertlangs.txt

# Get ParaRel data.
mkdir ${WORKDIR}/data/pararel
git clone git@github.com:yanaiela/pararel.git
mv pararel/data/pattern_data/graphs_json/* ${WORKDIR}/data/pararel
rm -r pararel

# Translate ParaRel.
mkdir -p ${WORKDIR}/data/multilingual/pararel
python translate_templates.py translate_folder \
--templates_folder ${WORKDIR}/data/pararel \
--output_folder ${WORKDIR}/data/multilingual/pararel \
--language_mapping_file mbertlangs.txt

# Clean templates in place.
cp -r ${WORKDIR}/data/multilingual/pararel ${WORKDIR}/data/multilingual/pararel_cleaned
python translate_templates.py clean_dir \
	--templates_folder ${WORKDIR}/data/multilingual/pararel_cleaned

# Copy templates and relations to an output folder
mkdir -p ${WORKDIR}/generated_datasets/mpararel_clean
python create_organized_output.py \
    --infolder ${WORKDIR}/data/multilingual \
	--templates_folder_name pararel_cleaned \
    --outfolder ${WORKDIR}/generated_datasets/mpararel_clean