#!/usr/bin/env bash
# @author: __rgalhama__

#Load global variables
if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../globals.sh
fi

#Export the pythonpath
export PYTHONPATH=$PYTHONPATH:${source_dir}


#Config file
if [[ "$#" -ne 2 ]]
then
#    config_file=$configs_dir'/config.json'
    echo "    Usage: ./prepare_env model config_file "
    exit
fi

MODEL="$1"
config_file="$2"
MODEL=`echo $MODEL |  tr '[:upper:]' '[:lower:]'`

#Prepare output directory structure
#Set subjects parameters
eval "$(python ${configs_dir}/config_loader.py ${config_file} "subjects" "VARS")"

## First level directory: LANGUAGE_AGEMIN_AGEMAX
output_dir_langage=$output_dir_base"/${shuffled}$lang"'_'"$agemin"'_'"$agemax"
if [ ! -d $output_dir_langage ]; then
    mkdir $output_dir_langage
fi
## Second level directory:
model_id=`eval "echo $(python ${configs_dir}/config_loader.py ${config_file} $MODEL "STRID")"`
output_dir=$output_dir_langage"/${model_id}/"
if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi
repr_dir=$output_dir

## Evaluation directories
### post params
model_id=`eval "echo $(python ${configs_dir}/config_loader.py ${config_file} "post" "STRID")"`
eval_dir=$output_dir"/${model_id}/"
if [ ! -d $eval_dir ]; then
    mkdir $eval_dir
fi
### aoa evaluation dir
eval "$(python ${configs_dir}/config_loader.py ${config_file} "extract_aoa" "VARS")"
aoa_eval_dir="${eval_dir}/evaluation_aoa_cdi_${proportion_cdi}_${measure_cdi}_${category}/"
if [ ! -d $aoa_eval_dir ]; then
    mkdir $aoa_eval_dir
fi

#Find corpus path, depending on subjects
corpus_file="${shuffled}${lang}_${agemin}_${agemax}_cds_stem.txt"
CORPUS=$data_dir_base'/'$corpus_file
corpus_freqs="${data_dir_base}/"${lang}"_"$agemin"_"$agemax"_cds_wordcounts.txt"

