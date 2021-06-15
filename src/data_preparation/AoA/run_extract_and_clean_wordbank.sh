#!/usr/bin/env bash
# @author: __rgalhama__


if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi

##### EXTRACT DATA ####
#######################
#1. call R script extract_AoA_wordbank.R (the following code has not been tested yet)
#  #!/usr/bin/Rscript
# source(extract_AoA_wordbank.R)
# quit(save='no')

#### CLEAN THE FILE ####
########################
#2. Clean
category="adjectives"
input_file="${cdi_dir}/data_for_study_2/it1/aoa_wordbank_eng_understands_prop0.5_${category}.csv"
output_file=`echo $input_file | sed 's/.csv/_clean.csv/'`

#Column: definition
cat $input_file | awk 'BEGIN{FS=";"}{sub(/ ?\(.*/,"",$4); sub(/ ?\/.*/,"",$4);print $0}' OFS=";" >  aux

#Column: uni_lemma
cat aux | awk 'BEGIN{FS=";"}{sub(/ ?\(.*/,"",$9); sub(/ ?\/.*/,"",$9);print $0}' OFS=";" >  $output_file

rm aux

###### COMBINE REPETITIONS ######
#################################
python combine_repetitions_wordbank.py $output_file

