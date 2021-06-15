#!/usr/bin/env bash
# @author: __rgalhama__

# Creates pairs and vocabulary
###############################

if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi

usageline="Usage: ./create_counts.sh config_file corpus_file output_dir"

if [ "$#" -ne 3 ]; then
    echo $usageline
    exit
fi

#Parse params
config_file=$1
CORPUS=$2
output_dir=$3

#Load options
OPTS=`eval "echo $(python $configs_dir/config_loader.py ${config_file} "counts" "CMDOPTS")"`
echo "Model: counts $OPTS"

#Create representations
echo "Counting co-occurrences..."
echo "Corpus to pairs..."
python $hyperwords_dir/hyperwords/corpus2pairs.py $OPTS  ${CORPUS} > $output_dir/pairs
echo "Pairs to counts..."
$hyperwords_dir/scripts/pairs2counts.sh $output_dir/pairs > $output_dir/counts
echo "Counts to vocab..."
python $hyperwords_dir/hyperwords/counts2vocab.py $output_dir/counts
echo "Done. Find the counts at "$output_dir

