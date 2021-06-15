#!/usr/bin/env bash
# @author: __rgalhama__


if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi


if [[ "$CONDA_PREFIX" != *"/wordrep"  ]]; then
    source activate wordrep
fi

###############################################################
#Evaluation
###############################################################
#
#    Usage: ./compute_index.sh <MODEL>  <model_dir> <config_file> <aoa_file> <output_dir>
#
#    The config file must specify the postprocessing options.
#
###############################################################


if [ "$#" -ne 5 ]; then
    echo "Usage: ./compute_index.sh <MODEL>  <model_dir>  <config_file> <aoa_file> <output_dir> "
    echo "Available models: ppmi / SVD / SGNS / GLOVE"
    echo ""
    exit
fi

MODEL="$1"
model_dir="$2"
config_file="$3"
task_path="$4"
output_dir="$5"

echo "Starting evaluation for "$MODEL" and config "${config_file}"...."
echo ""

repr=`echo $MODEL |  tr '[:upper:]' '[:lower:]'`
if [[ $repr == "ppmi" ]]
then
    repr="pmi"
fi



#Run
POST_OPTS=`eval "echo $(python $configs_dir/config_loader.py ${config_file} "post" "CMDOPTS")"`
EVAL_OPTS=`eval "echo $(python $configs_dir/config_loader.py ${config_file} "eval_aoa" "CMDOPTS")"`
eval "$(python $configs_dir/config_loader.py ${config_file} "subjects" "VARS")"
eval "$(python $configs_dir/config_loader.py ${config_file} "extract_aoa" "VARS")"
MODEL=`echo $MODEL |  tr '[:lower:]' '[:upper:]'`
echo "Computing index..."
#echo "python $source_dir/evaluation/indexs_from_reprs/repr2index.py $POST_OPTS $EVAL_OPTS $MODEL $model_dir$repr $task_path $output_dir"
python $source_dir/evaluation/indexs_from_reprs/repr2index.py $POST_OPTS $EVAL_OPTS $MODEL $model_dir$repr $task_path $output_dir
echo "Done. Results in ${output_dir}."
#awk 'BEGIN{FS=";" }{print $2}'| sort  | uniq | wc -l

#Deactivate virtual environment
source deactivate
