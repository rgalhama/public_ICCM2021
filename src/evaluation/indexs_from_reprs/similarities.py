"""
Author : Raquel G. Alhama
Desc:
"""
import sys, os
from os.path import join
import csv


def similarity_between_wordlists(lw1, lw2, representation, output_path):
    similarities={}
    fpath=join(output_path, "similarities_within_test.txt")
    with open(fpath, "w") as fh:
        for word1 in lw1:
            for word2 in lw2:
                similarity = representation.similarity(word1, word2)
                #Write to file
                line=word1 + " " + word2 + " " + str(similarity) + "\n"
                fh.write(line)

    #Sort output file
    # command = "cat "+fpath+" | sort -nk3r | sort -s -k1,1 > aux; mv aux " + fpath
    command = "cat "+fpath+" | sort -nk3r > aux; mv aux " + fpath
    os.system(command)

    return fpath

def compute_neighbours(wordlist, similarities_fpath, threshold, output_fpath):
    in_degree={}
    with open(similarities_fpath, "r") as fh:
        with open(output_fpath, 'w') as fo:
            for line in fh:
                word, neighbour, similarity = line.strip().split()
                if float(similarity) <= threshold:
                    #We are done
                    w = csv.writer(fo, delimiter=';')
                    w.writerows(in_degree.items())
                    return in_degree
                else:
                    if word != neighbour:
                        in_degree[word] = in_degree.get(word, 0) + 1


def find_n_threshold(word, representations, threshold, increment=500):
    """
    Find number of neighbours with cosine similarity >= threshold.
    Assumes that representations.closest is sorted (largest to smallest).
    :param word:
    :param representations:
    :param threshold:
    :param increment:
    :return:
    """
    n=increment
    neighbours =  [neigh[0] for neigh in representations.closest(word, n)]
    #Get more neighbours while we don't hit the threshold
    while neighbours[-1] >= threshold:
        n+=increment
        neighbours =  [neigh[0] for neigh in representations.closest(word, n)]
    #Reduce list to include only until threshold
    limit = len(neighbours) - 1
    while neighbours[limit] < threshold:
        limit -= 1
    return limit + 1


def compute_neighbours_from_repr(wordlist, representations, threshold, output_fpath):
    word_n={}
    for word in wordlist:
        n=find_n_threshold(word, representations,threshold)
        word_n[word]=n-1 #minus one, to remove the word itself

    with open(output_fpath, 'w') as fo:
        w = csv.writer(fo, delimiter=';')
        w.writerows(word_n.items())

    return word_n
