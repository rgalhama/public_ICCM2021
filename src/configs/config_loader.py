"""
Author : Raquel G. Alhama
Desc:
"""
import sys
import json

def load_config(fname):
    with open(fname, "r") as fh:
        params = json.load(fh)
    return params

def get_opts_model(params, model):
    return params[model]

def opts2cmdstr(opts):
    """Prints command line options in Bash."""
    cmdstr = ""
    for opt, val in opts.items():
        if val is not None:
            act=" --%s %s "%(opt,str(val))
            cmdstr+=act
    return cmdstr

def opts2w2vfcmdstr(opts):
    """Prints command line options in word2vecf format."""
    cmdstr = ""
    for opt, val in opts.items():
        if val is not None:
            act=" -%s %s "%(opt,str(val))
            cmdstr+=act
    return cmdstr

def opts2record(countopts, opts, post, sep=";"):

    header=""
    record = ""

    for options in [countopts, opts, post]:
        keys=sorted(options.keys())
        for opt in keys:
            header+=opt+sep
            val = options[opt]
            if val == "":
                val=True
            if val is None: #check none/null
                val=False
            record+=str(val)+sep

    #Output
    print(record)
    return header,record

def opts2strid(model, countopts, opts):
    """Prints a string, without spaces, with the model and parameters.
        To be used as an identifier of the model (e.g. for directory names)
        Format:  model_opt1_val1_opt2_val2
        For postprocessing params: opt1_val1_opt2_val2 (only post params)
        The first options should be the parameters of the count model. The rest are for the model itself.
    """

    cmdstr = model

    #Count params
    if model.lower() != "post":
        keys=sorted(countopts.keys())
        for opt in keys:
            val = countopts[opt]
            if val is not None:
                if str(val) is "":
                    act=opt
                else:
                    act="%s_%s"%(opt,str(val))
                cmdstr = cmdstr + "_" + act

    #Model params
    keys=sorted(opts.keys())
    for opt in keys:
        val = opts[opt]
        if val is not None:
            if str(val) is "":
                act=opt
            else:
                act="%s_%s"%(opt,str(val))
            cmdstr = cmdstr + "_" + act

    #Output
    print(cmdstr)
    return cmdstr

def get_opts_bash(opts):
    """Prints variable assignments in Bash."""
    for opt,value in opts.items():
        print("%s=\"%s\"\n"%(opt,value))

def json2bash(fname, model):
    params=load_config(fname)
    opts=get_opts_model(params, model)
    get_opts_bash(opts)

def tr_w2vf(opts):
    """

    :param opts:  counts+post dictionary of params
    :return: dictionary of params translated to w2vecf (also removing redundant ones)
    """
    tr_dict={
              "sub": "sample",\
              "thr": "min-count"
    }
    new_dict=opts.copy()
    for k in opts:
        if k in tr_dict.keys():
            new_dict[tr_dict[k]] = opts[k]
        new_dict.pop(k)


    return new_dict

def json2cmdopts(fname, model, w2vf=False):
    params=load_config(fname)
    opts=get_opts_model(params, model)

    #Extend and translate params if w2vecf models (sgns, cbow)
    if model.lower() == "sgns" or model.lower() == "cbow":
        count_params=get_opts_model(params, "counts")
        w2v_params=tr_w2vf(count_params)
        opts.update(w2v_params)

    if w2vf:
        cmdopts=opts2w2vfcmdstr(opts)
    else:
        cmdopts=opts2cmdstr(opts)

    print(cmdopts)

def json2strid(fname, model):
    params=load_config(fname)
    countopts=get_opts_model(params, "counts")
    opts=get_opts_model(params, model)
    strid=opts2strid(model, countopts, opts)
    return strid

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: config_loader.py config_file model [VARS/CMDOPTS/STRID]")
        exit(-1)
    if sys.argv[-1] == "VARS":
        json2bash(sys.argv[1], sys.argv[2])
    elif sys.argv[-1] == "CMDOPTS":
        json2cmdopts(sys.argv[1], sys.argv[2])
    elif sys.argv[-1] == "CMDOPTSW2VF":
        json2cmdopts(sys.argv[1], sys.argv[2], w2vf=True)
    elif sys.argv[-1] == "STRID":
        json2strid(sys.argv[1], sys.argv[2])
    else:
        print("Wrong option %s"%sys.argv[-1])