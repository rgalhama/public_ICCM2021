"""
Author : Raquel G. Alhama
Desc:
"""
import sys
import pandas as pd


def main(fname):
    df=pd.read_csv(fname, sep=";")
    sdf=df[["uni_lemma", "lexical_class", "aoa",]]
    gsdf=sdf.groupby(["uni_lemma","lexical_class"]).mean()
    newfname=fname.replace(".csv","_means.csv")
    gsdf.to_csv(newfname, sep=";")

if __name__=="__main__":

    args=sys.argv[1:]
    if len(args) != 1:
        print("Usage: combine_repetitions_wordbank.py <file>")
        exit(-1)
    main(*args)
