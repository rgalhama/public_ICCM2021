#Set this to your own directory structure
# @author: __rgalhama__

home_dir=`echo ~`
project_dir=`echo ~/Research/WordRep/WordRep_alhama/`

#Source directories
source_dir=${project_dir}"/04-source_code/src/"
toolkit_dir=${source_dir}"/toolkits/"
hyperwords_dir=${source_dir}"/toolkits/hyperwords_f5a01ea3e44c/"
w2vf_dir=${source_dir}"/toolkits/word2vecf/"

#Configurations
configs_dir=$source_dir"/configs/"


#Input data dir
data_dir_base=${project_dir}"/03-data/CHILDES_CDS/childes_db_extraction/"

#Tasks dirs (extend for particular studies)
cdi_dir="${project_dir}/03-data/AoA/wordbank/"

#Output base dir (to contain representation and evaluation)
output_dir_base=`echo ~/Data_Research/results_wordrep/`

