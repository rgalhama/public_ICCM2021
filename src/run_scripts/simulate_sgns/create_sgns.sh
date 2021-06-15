#!/usr/bin/env bash
# @author: __rgalhama__
#Load global vars
if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi
#


#Check correct usage
usageline="Usage: ./create_sgns.sh config_file corpus_file output_dir"
if [ "$#" -ne 3 ]; then
    echo $usageline
    exit
else
    config_file=$1
    CORPUS=$2
    output_dir=$3
fi
#


#Activate python environment
if [[ "$CONDA_PREFIX" != *"/wordrep"  ]]; then
    source activate wordrep
fi
#


#Create representations (adapted from hyperwords: corpus2sgns.sh)
echo "Computing SGNS ..."

#1. Counts: word-context pairs, vocabulary
COUNT_OPTS=`eval "echo $(python $configs_dir/config_loader.py ${config_file} "counts" "CMDOPTS")"`
python $hyperwords_dir/hyperwords/corpus2pairs.py ${COUNT_OPTS}  ${CORPUS} > $output_dir/pairs
$hyperwords_dir/scripts/pairs2counts.sh $output_dir/pairs > $output_dir/counts
python $hyperwords_dir/hyperwords/counts2vocab.py $output_dir/counts

#2. SGNS representation
SGNS_OPTS=`eval "echo $(python $configs_dir/config_loader.py ${config_file} "sgns" "CMDOPTSW2VF")"`
act_dir=`pwd`
cd $output_dir
$hyperwords_dir/word2vecf/word2vecf $COUNT_OPTS $SGNS_OPTS -train pairs -cvocab counts.contexts.vocab -wvocab counts.words.vocab -dumpcv sgns.contexts -output sgns.words

#3. Extract vectors, for further processing

python ${hyperwords_dir}/hyperwords/text2numpy.py $output_dir/sgns.words
python ${hyperwords_dir}/hyperwords/text2numpy.py $output_dir/sgns.contexts
POST_OPTS=`eval "echo $(python $configs_dir/config_loader.py ${config_file} "post" "CMDOPTS")"`
POST_OPTS=`eval "echo $(echo $POST_OPTS | grep -o "\-\-w\+c")"`
python ${hyperwords_dir}/hyperwords/sgns2text.py  $POST_OPTS $output_dir/sgns $output_dir/vectors.txt

cd $act_dir
echo "Done. Find the SGNS representation in $output_dir"


source deactivate