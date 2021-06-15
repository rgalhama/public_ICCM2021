#!/usr/bin/env bash

# @author: __rgalhama__
#Initialize
source init.sh


# Paths and params
lang="eng"
agemin=0
agemax=60
thresholds=(0.60 0.70 0.80 0.90)
proportion_cdi=0.5
task_path="${cdi_dir}/data/"
indextype="indegree_train"
categories=("nouns" "verbs")
models=("svd")
measure="produces"
here=`pwd`

# Methods
computeindexs () {
    measure=$1
    config_dir=$2
    model=$3

    echo $config_dir
    echo "Evaluating ${model}..."
    for config in `ls $config_dir`; do
       for thr in ${thresholds[@]} ; do
            for category in ${categories[@]}; do
                #Set category
                sed -i "s/\"category\".*$/\"category\"\:\ \"${category}\",/" ${config_dir}${config}
                #Set threshold
                sed -i "s/\"threshold\".*$/\"threshold\"\:\ \"${thr}\"/" ${config_dir}${config}
                #Set measure cdi
                sed -i "s/\"measure_cdi\".*$/\"measure_cdi\"\:\ \"${measure}\",/" ${config_dir}${config}
                #Set rest of vars
                source ${source_dir}/run_scripts/__prepare_env.sh $model ${config_dir}$config
                #AoA file
                aoa_file="${task_path}/aoa_wordbank_${lang}_${measure}_prop${proportion_cdi}_${category}_clean_means.csv"
                #Run index computation
                #index_fname="$aoa_eval_dir/${indextype}_th${th}.csv"
                #if [!-f $index_fname or `wc -l $index_fname` neq 282]; do
                ${source_dir}/run_scripts/evaluation/compute_index.sh $model $repr_dir ${config_dir}${config} $aoa_file $aoa_eval_dir
            done
       done
    done
}

computecorr () {
    for config in `ls $config_dir`; do
        for thr in ${thresholds[@]} ; do
            for category in ${categories[@]}; do

                    #Set category
                    sed -i "s/\"category\".*$/\"category\"\:\ \"${category}\",/" ${config_dir}${config}

                    #Prepare output folder
                    source ${source_dir}/run_scripts/__prepare_env.sh $model ${config_dir}/$config


                    #Merge with AoA
                    aoa_file="${aoapath}/aoa_wordbank_eng_${measure}_prop${proportion_cdi}_${category}_clean_means.csv"
                    corpusfreqs=`echo ${corpus_freqs}`
                    index_file=${aoa_eval_dir}/${index}_th${thr}.csv
                    python ${source_dir}/evaluation/stats/merge_idx_aoa.py ${index_file} $aoa_file "${corpusfreqs}" ${aoa_eval_dir}
                    datafname="merged_indegree_train_th${thr}aoa_wordbank_eng_${measure}_prop0.5_${category}_clean_means.csv"

                    #Compute index
                    python ${source_dir}/evaluation/stats/corr_index_aoa.py $model ${config_dir}/$config $thr ${aoa_eval_dir}/${datafname} ${aoa_eval_dir}

            done
        done
    done
}

# Procedure
##################################################

config_dir="${source_dir}/study_6_model_selection/configs/debugging/"

for model in "${models[@]}"; do

    cd $here

    #1. Generate config files
#    python generate_configs.py $model
#    mv list_of_confs_for_corrs_${model}.txt ${modelsdir}
#    mv configs/config*.json ${config_dir}

    #2. Train models
    for config in `ls $config_dir`; do
        ${source_dir}/run_scripts/create_representation.sh ${model} $config_dir${config}
        echo ""
    done

    #3. Compute indexs, for several similarity thresholds
    computeindexs "${measure}" ${config_dir} ${model}

    #4. Compute correlation between all the computed indexs and AoA data
    computecorr ${model}


    #5. Put all correlations together in one file
    cd $modelsdir
    #ToDo WARNING post_eig_1_neg_1
#    for pos in "${categories[@]}"; do
#        out_all_fn=correlations_${model}_${measure}_${pos}.csv
#        if [ -f $out_all_fn ]; then
#            rm $out_all_fn
#        fi
#        for m in `cat list_of_confs_for_corrs_${model}.txt`; do

#            #Results separated by threshold
#            for thr in "${thresholds[@]}"; do
#                pcorr_fn=$m/post_eig_1_neg_1/evaluation_aoa_cdi_0.5_${measure}_${pos}/params_corr_merged_indegree_train_th${thr}aoa_wordbank_eng_${measure}_prop0.5_${pos}_clean_means.csv
#                out_fn=correlations_${model}_thr${thr}_${measure}_${pos}.csv
#                if [ ! -f $out_fn ]; then
#                    head -n 1 $pcorr_fn > $out_fn
#                fi
#                if [ ! -f $out_all_fn ]; then
#                    head -n 1 $pcorr_fn > $out_all_fn
#                fi
#                tail -n 1 $pcorr_fn >> $out_fn
#                tail -n 1 $pcorr_fn >> $out_all_fn
#            done
#        done
#    cp $out_all_fn ${here}/results/
#    done
done
echo "Done"



##############################################################################################################
#Deactivate virtual environment
source deactivate

