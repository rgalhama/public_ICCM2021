#!/usr/bin/env bash

# @author: __rgalhama__
#scriptpath="$( cd "$(dirname "$0")" ; pwd -P )"

#First script that needs to be executed, to set environment vars
###################################################################


#Load global config
if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../globals.sh
fi

#Export the pythonpath
export PYTHONPATH=$PYTHONPATH:${source_dir}

#Activate the python interpreter
if [[ "$CONDA_PREFIX" != *"/wordrep"  ]]; then
#    source activate wordrep
#    conda activate wordrep
    source ~/Programas/miniconda3/envs/wordrep/bin/activate
fi

#Load local vars
source params.sh

