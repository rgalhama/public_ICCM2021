"""
Author : Raquel G. Alhama
Desc:
"""
from evaluation.indexs_from_reprs.similarities import *

"""
    Indexs of neighbourhood density computed from representations.
"""

def indegree_known(test_words_aoa, representations, threshold, output_fpath):
    """
    Computes neighbourhood density, only for words known so far.
    :param test_words_aoa:
    :param representations:
    :param threshold:
    :param output_fpath:
    :return:
    """
    raise NotImplementedError



def indegree_within_test(test_words, representations, threshold, output_fpath):

    word_n={}
    for word in test_words:
        total = 0
        n=find_n_threshold(word, representations,threshold)
        neighbours=representations.closest(word, n)

        #Count as neighbours only words we test
        for neighbour in neighbours:
            if neighbour[1] in test_words and neighbour[1] != word:
                total +=1

        word_n[word] = total

    #Write output file
    with open(output_fpath, 'w') as fo:
        w = csv.writer(fo, delimiter=';')
        w.writerows(word_n.items())


    return total

def indegree_train(wordlist, representations, threshold, output_fpath):

    #Compute
    word_n = compute_neighbours_with_heap(wordlist, representations, threshold)

    #Write output file
    with open(output_fpath, 'w') as fo:
        w = csv.writer(fo, delimiter=';')
        w.writerows(word_n.items())

    return word_n

