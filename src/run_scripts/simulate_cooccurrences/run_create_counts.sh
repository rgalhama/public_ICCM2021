#!/usr/bin/env bash
# @author: __rgalhama__

if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi

if [[ "$CONDA_PREFIX" != *"/wordrep" ]]; then
    source activate wordrep
fi

MODEL="counts"

##############################################
#
#   Create representations
#       Usage: ./run_create_counts [config_file]
#   If the config_file is not provided, the default one is used.
#
##############################################


if [[ -z "$1" ]]
then
    config_file=$configs_dir'/config.json'
    echo "Config file not provided. "
    echo "    Usage: ./run_create_counts [config_file] "
    echo "Running default configuration... ($config_file)"
else
    config_file="$1"
fi



##############################################
#Initialize parameters and environment vars  #
##############################################
source $source_dir/__prepare_env.sh $MODEL $config_file

##################################################################
#Create model
##################################################################

# Save configuration in representation dir
cp $config_file $output_dir

# Count co-occs
./create_counts.sh $config_file $CORPUS $output_dir

##################################################################
#Evaluate
##################################################################
# Hyperwords doesn't create a matrix for the term-term matrix, so it cannot be evaluated
#cd ..
#MODEL=`echo $MODEL |  tr '[:lower:]' '[:upper:]'`
#./evaluate.sh $MODEL $output_dir $config_file $eval_dir

#Deactivate virtual environment
source deactivate
