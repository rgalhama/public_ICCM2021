#!/usr/bin/env bash
# @author: __rgalhama__

############################################################################################
#
#    Creates representation
#        Usage: ./run_study_w2v [config_file]
#    If the config_file is not provided, the default one is used.
#
############################################################################################

#Set environment vars
if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi
#


MODEL="SGNS"
model="sgns"



# Initialize config
if [[ -z "$1" ]]
then
    config_file=$configs_dir'/config.json'
    echo "Config file not provided. "
    echo "    Usage: ./run_create_sgns [config_file] "

    read -p "Do you want to continue with the default config file? <y/N> " prompt
    if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]
    then
        echo "Running default configuration... ($config_file)"
    else
        exit 0
    fi
else
    config_file="$1"
fi
#


# Initialize environment (directories, etc)
source $source_dir/run_scripts/__prepare_env.sh $MODEL $config_file

# Save copy of configuration in representation dir
cp $config_file $output_dir



#Create model
##################################################################
$source_dir/run_scripts/simulate_sgns/create_sgns.sh $config_file $CORPUS $output_dir

##################################################################


