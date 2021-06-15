#!/usr/bin/env bash
# @author: __rgalhama__

#create representations from config file
# ./create_representations.sh <repr> <config_file>

#Params here or in command line
repr=$1
configfile=$2

if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../globals.sh
fi

source ${source_dir}/run_scripts/__prepare_env.sh $repr $configfile
if [ -f "${output_dir}/${repr}.words" ]; then
    echo "Model ${eval_dir} already exists. Skipping training."
else
    echo "Starting training for model ${repr_dir}..."
    $source_dir/run_scripts/simulate_${repr}/run_${repr}.sh $configfile
fi

