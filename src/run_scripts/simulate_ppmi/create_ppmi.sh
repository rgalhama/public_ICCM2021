#!/usr/bin/env bash

# @author: __rgalhama__

if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi

MODEL="ppmi"

usageline="Usage: ./create_ppmi.sh config_file corpus_file output_dir"

if [ "$#" -ne 3 ]; then
    echo $usageline
    exit
fi

#Parse params
config_file=$1
CORPUS=$2
output_dir=$3

#Load options
POST_OPTS=`eval "echo $(python $configs_dir/config_loader.py ${config_file} $MODEL "CMDOPTS")"`
echo "Model: $MODEL $POST_OPTS"

# Calculate PPMI matrix
echo "Computing $MODEL matrix..."
python $hyperwords_dir/hyperwords/counts2pmi.py $POST_OPTS $output_dir/counts $output_dir/pmi
echo "Done. Find the $MODEL matrix at $output_dir"