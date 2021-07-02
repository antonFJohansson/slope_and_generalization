import csv
import ast


def create_hyper_gen(file_name):
    ## File name of the text file with hyperparameters    
    
    all_paras = {}
    
    with open(file_name, 'r') as f:
        spamreader = csv.reader(f, delimiter=';')
        for row in spamreader:
            para_name = row[0]
            paras = row[1]
            if '#' in paras:
                paras = paras[0:paras.index('#')]
            paras = ast.literal_eval(paras)
            all_paras[para_name] = paras

    import itertools
    key_list = list(all_paras.keys())

    ## Fix there here
    all_items_list = []
    for k in key_list:
        all_items_list.append(all_paras[k])
   # return itertools.product(*all_items_list)
    for all_i in itertools.product(*all_items_list):
        yield(all_i)

import random
def convert_to_multiple_files(file_name):
    
    full_list = []
    for params in create_hyper_gen(file_name):
        full_list.append(params)
    
    random.shuffle(full_list)    
        
    max_param = 1
    sub_lists = [full_list[k:(k+max_param)] for k in range(0,len(full_list), max_param)]
    
    ## And here we can just create a list of lists?
    param_list = [[] for iii in range(len(params))]
    
    for idx,s_list in enumerate(sub_lists):
        txt_file = 'param' + str(idx) + '.txt'
        with open(txt_file, 'w') as f:
            f.write(str(s_list))

## And here we just load a file and see if it is ok?

#file_name = 'param0.txt'
def yield_sub_params(file_name):
    
    with open(file_name,'r') as f:
        txt = f.read()
        f = ast.literal_eval(txt)
    for it in f:
        yield it
    


file_name = 'largeParams.txt'
convert_to_multiple_files(file_name)






