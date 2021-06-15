"""
Author : Raquel G. Alhama
Desc:
"""
import json
import sys
from src.configs.config_loader import load_config, json2strid


wins=[1,2,3,4,5,7,10]
thrs=[10, 50, 100]
dyns=[None, ""]
sizes=[100]#tried 500 but gives lots of nan
negatives=[15] # only for sgns; tried 0 gives lots of nan
negs=[1,15] #only for svd
#simthrs=[0.6, 0.7, 0.8, 0.9] Not generate in configs but only during evaluation

def generate_config_files(baseconf, repr):
    confnumbr=1
    lines=""
    for actconf in generate_configs(baseconf, repr):
        newf="configs/config%i.json"%confnumbr
        with open(newf, "w") as fn:
            json.dump(actconf, fn, indent=4, sort_keys=True)
        name=json2strid(newf, repr)
        lines+=name+"\n"
        confnumbr+=1

    #Write list of model ids
    with open("list_of_confs_for_corrs_%s.txt"%repr, "w") as fn:
        fn.write(lines)

def generate_configs(baseconfig_f, repr):
    actconf=load_config(baseconfig_f)
    for win in wins:
        for thr in thrs:
            for dyn in dyns:
                for size in sizes:
                    actconf['counts']['win'] = win
                    actconf['counts']['thr'] = thr
                    actconf['counts']['dyn'] = dyn
                    actconf['svd']['dim'] = size
                    actconf['sgns']['size'] = size
                    if repr == 'sgns':
                        for negative in negatives:
                            actconf['sgns']['negative'] = negative
                            yield actconf
                    elif repr == 'svd':
                        for neg in negs:
                            actconf['svd']['neg'] = neg
                            yield actconf



if __name__ == "__main__":
    baseconfig_f="configs/base/base.json"
    model=sys.argv[1]
    print(model)
    generate_config_files(baseconfig_f, model)
