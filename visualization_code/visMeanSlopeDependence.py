
## Code used to obtain Figure 4 and Figure 5 in the article

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


save_figure = True
ds_list = ['fc', 'MNIST', 'KMNIST', 'FMNIST'] ## The datasets to plot, 'fc', 'KMNIST', 'MNIST', 'FMNIST'
conv_list = ['False'] ## Whether a convolutional model was used or not
depth_list = [1,2,3] ## Which number of layers to plot
width_list = [25, 50, 100, 200] ## Which amount of neurons to plot
sample_list = [0,1,2,3,4] ## Which samples to include in the plot

model_tuple_store = []

first_folder = 'res'
base_folder_list = [f for f in os.listdir(first_folder)]

all_info_list = []


full_info_dict = {}

for fold in base_folder_list:
    
    ds, cms, d, w, sample = re.findall(r'Experiment \d*([A-Za-z]*)_([A-Za-z]*)_(\d*)_(\d*)_(\d*)', fold)[0]
    
    if ds in ds_list and cms in conv_list and int(d) in depth_list and int(w) in width_list and int(sample) in sample_list:
        #print(sample)
        if not (ds, cms, int(d), int(w)) in full_info_dict.keys():
            full_info_dict[(ds, cms, int(d), int(w))] = []
        
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
        #print(sample)
        full_info_dict[(ds, cms, int(d), int(w))].append((info_list, all_info_dict))


    
full_res_dict = {}

for k in full_info_dict.keys():
    
    full_res_dict[k] = {'full_list': [],
                        'mean_list': []}
    
    for li, info_dict in full_info_dict[k]:
    
        
        full_res_list = []
        for e, p in li:
        
            df = pd.read_csv(p, sep = ';')
            curr_sing_vals = np.mean(df['max_sing1'].to_numpy())
            full_res_list.append(curr_sing_vals)
        
        ## More properties can be added here
        best_epoch = np.argmin(np.array(info_dict['testloss']))
        
        full_res_dict[k]['full_list'].append((best_epoch, full_res_list, info_dict['testloss'][best_epoch], info_dict['trainloss'][best_epoch]))
        
        full_res_dict[k]['mean_list'].append(full_res_list[best_epoch])
    
    


plot_list = []
print_list = []
for co in conv_list:
    for d in depth_list:
        for w in width_list:
            plot_list.append(('{}', co, d, w))
            print_list.append((d,w))

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

#figure, ax = plt.subplots(1,1, figsize=(20,10))
figure, ax = plt.subplots(1,1, figsize=(20,8))

color_dict = {'fc': 'r',
              'MNIST': 'k',
              'FMNIST': 'g',
              'KMNIST': 'b'}

name_dict = {'fc': 'Forest cover',
              'MNIST': 'MNIST',
              'FMNIST': 'FashionMNIST',
              'KMNIST': 'KMNIST'}


for ds in ds_list:
    mean_pred = []
    std_pred = []
    
    for it in plot_list:
        i1 = it[0].format(ds)
        i2 = it[1]
        i3 = it[2]
        i4 = it[3]
        its = (i1,i2,i3,i4)
        

        #print(it,'testloss', full_res_dict[it]['full_list'][0][2],full_res_dict[it]['full_list'][1][2],full_res_dict[it]['full_list'][2][2])
        #print(it, full_res_dict[it]['full_list'][0][3],full_res_dict[it]['full_list'][1][3],full_res_dict[it]['full_list'][2][3])
        mp = np.mean(np.array(full_res_dict[its]['mean_list']))
        sp = np.std(np.array(full_res_dict[its]['mean_list']))
        mean_pred.append(mp)
        std_pred.append(sp)
        
    
    
    ax.errorbar([k for k in range(len(mean_pred))], mean_pred, std_pred, label=name_dict[ds], color=color_dict[ds])




#ax.set_ylim(0,10)
ax.set_ylim(2,8)
ax.grid(axis='y')
x = [k for k in range(len(print_list))]
labels = print_list
#plt.plot(x,y, 'r')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Slope')
ax.set_xlabel('Network size (Number of hidden layers, Width of hidden layers)')
ax.legend()
ax.set_title('Evolution of slope mean for fully connected networks')

if save_figure:
    plt.savefig('mean_slope_dependence.png')

