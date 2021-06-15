
# @author: __rgalhama__
#PARAMS
lang="eng"
agemin=0
agemax=60
scriptdir=`pwd`
proportion_cdi=0.5
pos_l=("nouns" "verbs")
index="indegree_train"
#Distance threshold we want to look into atm
th="0.70" #threshold we are interested in; write with 2 decimals
measures_cdi=("understands" "produces")
#Shuffled words (for baseline); set to empty string if false, otherwise set to "shuffled"
shuffled="" #empty in most cases


#PATHS
modelsdir="${output_dir_base}/${shuffled}${lang}_${agemin}_${agemax}/"
aoapath="${project_dir}/03-data/AoA/wordbank/"
corpus_freqs="${data_dir_base}/${lang}_${agemin}_${agemax}_cds_wordcounts.txt"
