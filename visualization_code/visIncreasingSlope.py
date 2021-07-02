

## Code to obtain Figure 1 in the article

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


import matplotlib

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random

def retrieve_all_info(txt):
    
    all_info_dict = {}
    
    all_keys = ['model','dataset', 'numepochs','trainacc', 'testacc',
                'trainloss', 'testloss', 'batchsize', 'corrrate',
                'numestimates']
    ast_list = ['trainacc', 'testacc',
                'trainloss', 'testloss']
    
    for k in all_keys:
        ind1 = txt.find(k + 'start:')
        ind2 = txt.find(':' + k + 'end')
        param = txt[(ind1 + len(k + 'start:')):ind2]
        
        if k in ast_list:
            all_info_dict[k] = ast.literal_eval(param)
        else:
            all_info_dict[k] = param
    
    return all_info_dict


model_tuple_store = []

first_folder = 'res'
base_folder_list = [f for f in os.listdir(first_folder)]

all_info_list = []

#ds_list = ['fc']
ds_list = ['fc', 'MNIST', 'KMNIST', 'FMNIST']
conv_list = ['False']
depth_list = [1,2,3]
width_list = [25, 50, 100, 200]
sample_list = [0,1,2,3,4]

full_info_dict = {}

for fold in base_folder_list:
    
    ds, cms, d, w, sample = re.findall(r'Experiment \d*([A-Za-z]*)_([A-Za-z]*)_(\d*)_(\d*)_(\d*)', fold)[0]
    
    if ds in ds_list and cms in conv_list and int(d) in depth_list and int(w) in width_list and int(sample) in sample_list:
        #print(sample)
        if not (ds, cms, int(d), int(w)) in full_info_dict.keys():
            full_info_dict[(ds, cm, int(d), int(w))] = []
        
        info_list = []
    
        for tx in os.listdir(os.path.join(first_folder, fold)):
            
            if not 'info' in tx and not '.png' in tx and not '.pt' in tx:
                
                        full_path = os.path.join(first_folder, fold, tx)
                        ind1 = tx.find('e_')
                        ind2 = tx.find('.txt')
                        #print(tx[ind1:ind2])
                        epoch = float(tx[(ind1 + 2):ind2])
                        
                        info_list.append((epoch, full_path))
                        
        with open(os.path.join(first_folder, fold, 'info.txt'), 'r') as fi:
                txt = fi.read()
                all_info_dict = retrieve_all_info(txt)
                        
        info_list = sorted(info_list, key=lambda x: x[0])
        full_info_dict[(ds, cms, int(d), int(w))].append((info_list, all_info_dict))


    
full_res_dict = {}

for k in full_info_dict.keys():
    
    full_res_dict[k] = {'full_list': [],
                        'mean_list': []}
    
    for li, info_dict in full_info_dict[k]:
        #print(info_dict['testloss'][20])
        
        full_res_list = []
        for e, p in li:

            df = pd.read_csv(p, sep = ';')
            curr_sing_vals = np.mean(df['max_sing'].to_numpy())
            full_res_list.append(curr_sing_vals)
        
        ## More properties can be added here
        best_epoch = np.argmin(np.array(info_dict['testloss']))
        full_res_dict[k]['full_list'].append((best_epoch, full_res_list, info_dict['testloss'][best_epoch], info_dict['trainloss'][best_epoch]))
        
        full_res_dict[k]['mean_list'].append(full_res_list[best_epoch])
    

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

figure, ax = plt.subplots(1,1, figsize=(20,10))


all_plot_vals = []
for k in full_res_dict.keys():
    for it in full_res_dict[k]['full_list']:
        plot_vals = it[1]
        ax.plot(plot_vals, 'b', alpha = 0.2)
        
    
all_plot_vals = np.array(all_plot_vals)
all_plot_vals = np.mean(all_plot_vals, axis = 0)
ax.plot(all_plot_vals, 'b', label = 'Mean (fully connected)', linewidth = 3)
    
ax.grid(axis='y')
ax.set_xlabel('Epoch')
ax.set_ylabel('slope')
ax.set_yscale('log')
ax.legend()

