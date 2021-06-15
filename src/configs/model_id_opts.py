"""
Author : Raquel G. Alhama
Desc:
"""
def strid_to_opts(strid):
    """
    Given model id as string, extract parameter dictionary.
    Reverse of config_loader.opts2strid
    :param strid:
    :return:
    """


    raise NotImplementedError

    #Method not finished
    parts = strid.split("_")
    param_keys=",".split("thr,win,dim,neg,dim,size,eig,neg,dyn,cds") #finish
    d={}
    for i,part in enumerate(parts):
        if part == 'post':
            pass
        elif part in param_keys:
            if i<len(parts) and not parts[i+1] not in param_keys:
                k=part
                v=parts[i+1]
                d[k]=v
            else: #key without value
                k=part
                v=1
                d[k]=v
        else: #value
            pass

    return d
   # for p in parts: