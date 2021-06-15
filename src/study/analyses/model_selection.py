"""
Author : Raquel G. Alhama
Desc:
"""
from os.path import join
from analyze import load_all_data



def best_model(df):
    # idxmax=df.loc[df['pearsonr'].idxmax()]
    df['absr']=df['pearsonr'].abs()
    return df[df['absr']==df['absr'].max()]

def worse_model(df):
    # idxmax=df.loc[df['pearsonr'].idxmax()]
    df['absr']=df['pearsonr'].abs()
    return df[df['absr']==df['absr'].min()]


def main (resultspath):

    #1. Load data
    names = ["sgns_produces_nouns", "sgns_produces_verbs", "svd_produces_nouns", "svd_produces_verbs"]
    alldfs = load_all_data(resultspath)

    #Remove NaNs (they result from zero variance datasets).
    alldata={}
    for i,df in enumerate(alldfs):
        newdf=df.dropna()
        alldata[names[i]]=newdf

    #2. Find best and worse model params
    for name,df in alldata.items():
        bdf=best_model(df)
        bdf.to_csv(join(resultspath, "best_model_%s.csv"%name), sep=";")
        wdf=worse_model(df)
        wdf.to_csv(join(resultspath, "worst_model_%s.csv"%name), sep=";")

if __name__ == "__main__":

    resultspath="../results/"
    main(resultspath)
