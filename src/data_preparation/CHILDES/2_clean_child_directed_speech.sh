#!/usr/bin/env bash
# @author: __rgalhama__

#Set globals (such as project_dir)
if [[ "$source_dir" == "" ]]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    source ${DIR}/../../globals.sh
fi

#Path
path="${project_dir}/03-data/CHILDES_CDS/childes_db_extraction/"
cd $path


#Lowercase (only for utterances; stems are already lowercased, except proper nouns
#fname=eng_0_60_cds_utterances.txt
#cat $fname | tr '[:upper:]' '[:lower:]' > lowercased.txt
#mv lowercased.txt $fname

#Get word counts 
#wc_fname=eng_0_60_cds_wordcounts.txt
#cat $fname | tr " " "\n"| sort | uniq -c > freq.txt
#cat freq.txt | sort -nr -k1 > $wc_fname
#cat $wc_fname | sed 's/^ *//g' > freq.txt
#mv freq.txt $wc_fname

#
fname=eng_0_60_cds_stem.txt
wc_fname=eng_0_60_cds_wordcounts_stem.txt
cat $fname | tr " " "\n"| sort | uniq -c > freq.txt
cat freq.txt | sort -nr -k1 > $wc_fname
cat $wc_fname | sed 's/^ *//g' > freq.txt
mv freq.txt $wc_fname

echo "Total number of word types:"
wc -l $wc_fname
echo "Total number of word tokens:"
wc -w $fname

