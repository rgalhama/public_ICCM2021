"""
Author : Raquel G. Alhama
Desc:
"""
import sys, os, inspect
from os.path import join
from docopt import docopt
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#Add source folder to the path:
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, os.pardir, os.pardir)
if MAIN_FOLDER not in sys.path:
    sys.path.insert(0, MAIN_FOLDER)
from evaluation.indexs_from_reprs.indexs import *
import toolkits.hyperwords_f5a01ea3e44c.hyperwords.representations.representation_factory as reprf




def get_fname_index(index, threshold):
    return "%s_th%.2f.csv"%(index, float(threshold))

def get_fname_eval_output(prefix, index, threshold, log_transformed, extension):
    fname="%s_%s_th%.2f"%(prefix, index, float(threshold))
    if log_transformed:
        fname = fname + "_log"
    return fname+extension


def eval_against_aoa(word_measure, word_aoa, path, args):
    #Merge data
    word_aoa.rename(columns={"uni_lemma":"word"}, inplace=True)
    mdf = pd.merge(word_measure, word_aoa, how="inner", on="word")
    cols=mdf[['measure','aoa']]
    X=mdf['measure']
    Y=mdf["aoa"]


    #Pearson r
    #pearsonr=cols['aoa'].corr(cols['measure'], method='pearson')
    pearsonr, pearson_pval = stats.pearsonr(X, Y)

    #Spearman rho (ranked correlation)
    # spearman=cols['aoa'].corr(cols['measure'], method='spearman')
    spearman_rho, spearman_pval = stats.spearmanr(X, Y)

    #Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(list(X),list(Y))
    line = slope*X+intercept

    #Plot
    scatter_fname = get_fname_eval_output("eval_aoa_scatter", args["--index"], args["--threshold"], args["--log"], ".png")
    plt.scatter(cols['measure'], cols['aoa'])
    plt.xlabel("model")
    plt.ylabel("AoA")
    plt.plot(X,line,'r-',X,Y,'o')
    # plt.show()
    plt.savefig(join(path, scatter_fname))

    #Coefficient of determination?
    #...todo

    #Write output
    eval_fname = get_fname_eval_output("eval_aoa", args["--index"], args["--threshold"], args["--log"], ".txt")
    with open(join(path,eval_fname), "w") as fh:
        fh.write("Pearson r: %.3f pval:%.3f\n"%(pearsonr,pearson_pval))
        fh.write("Spearman rho: %.3f pval:%.3f\n"%(spearman_rho, spearman_pval))
        fh.write("Regression:\n")
        fh.write("\tslope:     %.3f\n"%slope)
        fh.write("\tintercept: %.3f\n"%intercept)
        fh.write("\tr_value:   %.3f\n"%r_value)
        fh.write("\tp_value:   %.3f\n"%p_value)
        fh.write("\tstd_err:   %.3f\n"%std_err)

    return

def filter_vocab(test_words, train_words):
    a=set(test_words)
    b=set(train_words)
    return a.intersection(b)

def read_test_set(path):
    df = pd.read_csv(path, sep=";")
    words = df.uni_lemma
    return list(words)

def read_word_measure_to_dict(fpath):
    with open(fpath, mode='r') as fh:
        reader = csv.reader(fh, delimiter=";")
        word_measure = dict((rows[0],rows[1]) for rows in reader)
    return word_measure



def main():
    args = docopt("""
    Usage:
        repr2index.py [options] <representation> <representation_path> <task_path> <output_path>
    
    Options:
        
        --neg NUM       Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c           Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM       Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
        
        --index STR   Measure to compute. Options: indegree_test, indegree_train
        --threshold NUM Minimum similarity for semantic networks. Must be between 0 and 1.
        
    """)

    #Load children's known words (test data)
    test_data = read_test_set(args['<task_path>'])
    #Load and postprocess representations
    representation = reprf.create_representation(args)

    #Filter words that are not in child directed speech
    filtered_test_data = filter_vocab(test_data, representation.iw)

    #Compute network index that will be evaluated
    eval_type = args["--index"].lower()
    index_output_fpath=join(args["<output_path>"], get_fname_index(args["--index"], args["--threshold"]))


    if eval_type == "indegree_test":
        indegree_within_test(filtered_test_data, representation, float(args['--threshold']), index_output_fpath)
    elif eval_type == "indegree_train":
        indegree_train(filtered_test_data, representation, float(args['--threshold']),index_output_fpath)
    else:
        raise Exception("Unknown measure option: %s"%eval_type)

    print("Index computed! Find it at %s"%index_output_fpath)

if __name__ == '__main__':
    main()
#python repr2index.py --neg 1 --eig 1 --threshold 0.7 --index indegree_test SGNS /home/rgalhama/Data_Research/results_wordrep//eng_0_60/sgns_dyn_thr_100_win_3_negative_1_pow_1_size_500/sgns /home/rgalhama/Research/WordRep/WordRep_alhama//03-data/AoA/wordbank//data_for_study_2/it1//aoa_wordbank_eng_understands_prop0.5_nouns_clean_means.csv /home/rgalhama/Data_Research/results_wordrep//eng_0_60/sgns_dyn_thr_100_win_3_negative_1_pow_1_size_500//post_eig_1_neg_1//evaluation_aoa_cdi_0.5_understands_nouns/
