# About 
Parameter reference list.
Many of these parameters are further explained in Levy et al., 2015.

These parameters can be adjusted using a configuration file, with the same format as the example file config.json.

**Important:** 
To add optional parameters to the json file, use "" as value. 
To remove them, set a null value.


# Parameters 

## Subjects 
lan:          STR    Language of the corpus (3-letter code)
agemin:       NUM    Minimum age in months
agemax:       NUM    Maximum age in months

## Counts. These options are used by all representations, when creating word-context pairs (corpus2pairs.py)
win:          NUM    Window size [default: 2]
thr:          NUM    Minimal word count for being in the vocabulary. Co-occurrence counts are only updated for words that are part of the vocabulary (those below thr are not dimensions in the matrix). [default: 100]
sub:          NUM    Subsampling threshold, to reduce impact of frequent words. It deletes the words with a certain probability that depends on this parameter (see Levy&Goldberg,2015) [default: 0]
del:                 If enabled, delete the words before creating the window. TODO check whether it concerns only words below thr or also subsampling.
dyn:                 Dynamic context windows: the window size is sampled.
pos:                 Positional contexts (??) #TODO see if this really belongs here
              
## PPMI 
cds:          NUM    Context distribution smoothing [default: 1.0]. Reccommended in Levy et al. (2015): 0.75.

## SVD 
dim           NUM    Dimensionality of eigenvectors [default: 500]
neg           NUM    Shifted PMI (equivalent to "negative" in SGNS). Subtracts its log from PMI. Bigger than 1 is detrimental for SVD (Levy et al. 2015) [default: 1]


## word2vecf: SGNS, SG, CBOW 

# The following parameters from word2vecf overlap with hyperwords toolkit:  
# window-->win; sample-->sub; min-count-->thr; pos->pos; 
# Roughly equivalent, in other models: negative->neg(roughly); pow->cds; size->dim)
# The following params will be taken from equivalent ones in "count" params dict:
(sub) sample        NUM    <float> set threshold for occurrence of words and contexts. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 0 (off), useful value in the original word2vec was 1e-5
(thr) min-count     NUM    <int> this will discard words that appear less than <int> times; default is 5
# These need to be specified in "sgns" params dict, in config.json file
size          NUM    <int> size of word vectors; equivalent to dim: dim [default: 100]
pow           NUM    <float> Equivalent to cds. Context distribution smoothing.
negative      NUM    <int> number of negative examples; default is 15, common values are 5 - 10 (0 = not used). Contrary to SVD, numerous negative samples (neg>1) are benefitial. 
alpha         NUM    <float> set the starting learning rate; default is 0.025
iters         NUM    <int> Perform i iterations over the data [default: 1]

# These are actually w2vec, not w2vecf!
#window        NUM    <int> window size  THIS WAS W2VEC, NOT W2VECF!
#cbow          NUM    1 if cbow; 0 if skipgram [default: 0]
#hs            NUM    1: use Hierarchical Softmax  (0 = not used) [default: 1]
#pos           NUM    1: Include sequence position information


## Post-processing parameters (modify vectors). 
# Used when converting vectors to text (w+c, eig) and/or in evaluation (w+c, eig, neg)
### Representation
w+c                  Add context vector in addition to word vector (not applicable to PPMI).
eig           NUM    Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
neg           NUM    As post, this param is only used for PPMI (in ws_eval). Applies shifted PMI [default: 1]

### Evaluation on Age of Acquisition
index         STR    Characterization of the semantic network used for the study (options: indegree_test, indegree_train, ...).
threshold     NUM    Minimum similarity for words to be considered neighbours (connected to target).
log                  Transform the measure into log.
proportion_cdi       Minimum proportion needed to consider a word is known at a certain age. Used when extracting the data.
measure_cdi    STR   Measure to assess when words are known by a child ("produces" or "understands")
category            STR   Indicates if the evaluation is restricted to certain PoS (e.g. "nouns", or "nouns_adjs"). Empty string means "all". The folder containing the output of the evaluation will be named accordingly.
