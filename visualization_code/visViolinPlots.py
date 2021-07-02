

## Used to obtain Figure 2 in the article

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

import seaborn as sns

import matplotlib

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random



def retrieve_all_info(txt):
    
    ## Function to retrieve the info stored from the results
    
    all_info_dict = {}
    
    all_keys = ['model','dataset', 'numepochs','trainacc', 'testacc',
                'trainloss', 'testloss', 'batchsize',
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
before_training = False ## To obtain violinplot before or after training
dataset_plot = 'fc' ## The dataset to plot, 'fc', 'KMNIST', 'MNIST', 'FMNIST'
conv_model = 'False' ## Whether a convolutional model was used or not
depth_list = [1,2,3] ## Which number of layers to plot
width_list = [25, 50, 100, 200] ## Which amount of neurons to plot
sample_list = [0]#,1,2,3,4] ## Which run to plot


first_folder = 'res' ## Where the results are stored
base_folder_list = [f for f in os.listdir(first_folder)]

model_tuple_store = []

all_info_list = []
ds_list = [dataset_plot]
conv_list = [conv_model]

full_info_dict = {}

## Obtain the correct paths for the considered models
for fold in base_folder_list:
    
    ds, cms, d, w, sample = re.findall(r'Experiment \d*([A-Za-z]*)_([A-Za-z]*)_(\d*)_(\d*)_(\d*)', fold)[0]
    
    if ds in ds_list and cms in conv_list and int(d) in depth_list and int(w) in width_list and int(sample) in sample_list:
        
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
        full_info_dict[(ds, cms, int(d), int(w))].append((info_list, all_info_dict))


    
full_res_dict = {}

## Store the info for the considered models
for k in full_info_dict.keys():
    
    full_res_dict[k] = {'full_list': [],
                        'mean_list': []}
    
    for li, info_dict in full_info_dict[k]:
        
        
        full_res_list = []
        for e, p in li:
            df = pd.read_csv(p, sep = ';')
            curr_sing_vals = df['max_sing1'].to_numpy().tolist()
            full_res_list.append(curr_sing_vals)
        
        ## More properties can be added here
        best_epoch = np.argmin(np.array(info_dict['testloss']))
        
        full_res_dict[k]['full_list'].append((best_epoch, full_res_list, info_dict['testloss'][best_epoch], info_dict['trainloss'][best_epoch]))
        full_res_dict[k]['mean_list'].append(full_res_list[best_epoch])
    
all_sing_vals = []
all_keys = []

## Create the dataframe needed to create the violinplot
k = [dataset_plot, conv_model]
for d in depth_list:
    for w in width_list:
        curr_key = tuple(k + [d,w])
        best_ind = full_res_dict[curr_key]['full_list'][0][0]
        if before_training:
            sing_vals = full_res_dict[curr_key]['full_list'][0][1][0]
        else:
            sing_vals = full_res_dict[curr_key]['full_list'][0][1][best_ind]
        all_sing_vals.extend(sing_vals)
        all_keys.extend(len(sing_vals)*[(d,w)])

df = pd.DataFrame(list(zip(all_sing_vals, all_keys)),
               columns =['slope', 'Network size (Number of hidden layers, Width of hidden layers)'])

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
figure, ax1 = plt.subplots(1,1,sharex = True, figsize= (20,8))

ax = sns.violinplot(x="Network size (Number of hidden layers, Width of hidden layers)", y="slope", data=df, color = (0.349, 0.49, 0.749))#, color='b')

ax1 = ax


ax1.grid(axis='y')
ax1.set_title('Slope distribution before training')

if save_figure:
    plt.savefig('slope_training_{}_208_before_{}.png'.format(dataset_plot, before_training))


