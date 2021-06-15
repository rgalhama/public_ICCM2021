#!/usr/bin/env bash

# @author: __rgalhama__

if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi

if [[ "$CONDA_PREFIX" != *"/wordrep" ]]; then
    source activate wordrep
fi

MODEL="PPMI"

##############################################
#
#   Create representations with PPMI
#       Usage: ./run_study_ppmi [config_file]
#   If the config_file is not provided, the default one is used.
#
##############################################

if [[ -z "$1" ]]
then
    config_file=$configs_dir'/config.json'
    echo "Config file not provided. "
    #echo "    Usage: ./run_create_ppmi [config_file] "
    echo "Running default configuration... ($config_file)"
else
    config_file="$1"
fi

##############################################
#Initialize parameters and environment vars  #
##############################################

source $source_dir/run_scripts/__prepare_env.sh $MODEL $config_file

##################################################################
#Create model
##################################################################

# Save configuration in representation dir
cp $config_file $output_dir

# Count co-occurrences
$source_dir/run_scripts/simulate_cooccurrences/create_counts.sh $config_file $CORPUS $output_dir

#Create PMI
$source_dir/run_scripts/simulate_ppmi/create_ppmi.sh $config_file $CORPUS $output_dir


##################################################################
#Evaluate
##################################################################
#MODEL=`echo $MODEL |  tr '[:upper:]' '[:lower:]'`
#repr_dir=$output_dir
#cd ..
#echo "Evaluating ${MODEL} ${repr_dir} ${config_file} ${eval_dir}"
#./evaluate.sh $MODEL $repr_dir $config_file $eval_dir
#nmbr=`echo $outcome | grep 'result:' | sed 's/result: //'`
# > $output_dir/evaluation.txt

#Deactivate virtual environment
source deactivate
