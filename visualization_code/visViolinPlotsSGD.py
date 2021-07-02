
## Code used to obtain Figure 3 in the article

import re
import os
import matplotlib.pyplot as plt
import numpy as np
import ast

import pandas as pd
import seaborn as sns

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
dataset_plot = 'MNIST' ## The dataset to plot, 'fc', 'KMNIST', 'MNIST', 'FMNIST'
conv_model = 'False' ## Whether a convolutional model was used or not
depth_list = [1,2,3] ## Which number of layers to plot
width_list = [25, 50, 100, 200] ## Which amount of neurons to plot
sample_list_seed = np.random.choice([0,1,2,3,4], 2, replace = False).tolist() ## Which of the two seeds to plot

model_tuple_store = []

first_folder = 'res'
base_folder_list = [f for f in os.listdir(first_folder)]

all_info_list = []


ds_list = [dataset_plot]
conv_list = [conv_model]
sample_list = [sample_list_seed[0]]

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
        full_info_dict[(ds, cms, int(d), int(w))].append((info_list, all_info_dict))


    
full_res_dict = {}

for k in full_info_dict.keys():
    
    full_res_dict[k] = {'full_list': [],
                        'mean_list': []}
    
    for li, info_dict in full_info_dict[k]:
        
        
        full_res_list = []
        for e, p in li:
            df = pd.read_csv(p, sep = ';')
            curr_sing_vals = df['max_sing1'].to_numpy().tolist()
            full_res_list.append(curr_sing_vals)
        
        ## Can add more properties here
        best_epoch = np.argmin(np.array(info_dict['testloss']))
        
        full_res_dict[k]['full_list'].append((best_epoch, full_res_list, info_dict['testloss'][best_epoch], info_dict['trainloss'][best_epoch]))
        
        full_res_dict[k]['mean_list'].append(full_res_list[best_epoch])
    
all_sing_vals = []
all_keys = []
all_bools = []

k = [dataset_plot, conv_model]
for d in depth_list:
    for w in width_list:
        curr_key = tuple(k + [d,w])
        best_ind = full_res_dict[curr_key]['full_list'][0][0]
        sing_vals = full_res_dict[curr_key]['full_list'][0][1][best_ind] # best_ind instead of 0
        all_sing_vals.extend(sing_vals)
        all_keys.extend(len(sing_vals)*[(d,w)])
        all_bools.extend(len(sing_vals)*['Seed 1'])





###################################################
###################################################3
#######################################################


model_tuple_store = []

first_folder = 'res'
#base_folder_list = ['diff_class_400_3_28','diff_class_400_3_56',
#                    'diff_class_401_3_28','diff_class_401_3_56']
all_info_list = []



sample_list = [sample_list_seed[1]]

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
        full_info_dict[(ds, cms, int(d), int(w))].append((info_list, all_info_dict))


    
full_res_dict = {}

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
    
k = [dataset_plot, conv_model]
for d in depth_list:
    for w in width_list:
        curr_key = tuple(k + [d,w])
        best_ind = full_res_dict[curr_key]['full_list'][0][0]
        sing_vals = full_res_dict[curr_key]['full_list'][0][1][best_ind] # best_ind instead of 0
        all_sing_vals.extend(sing_vals)
        all_keys.extend(len(sing_vals)*[(d,w)])
        all_bools.extend(len(sing_vals)*['Seed 2'])









## Create a dataframe with the required info for violinplots

df = pd.DataFrame(list(zip(all_sing_vals, all_keys, all_bools)),
               columns =['Slope', 'Network size (Number of hidden layers, Width of hidden layers)', 'Random seed'])



font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

plt.figure(figsize=(20,8))
ax = sns.violinplot(x="Network size (Number of hidden layers, Width of hidden layers)", y="Slope", hue = 'Random seed', data=df, palette="muted", split = True)#, color='b')


ax.grid(axis='y')


ax.set_title('Effect of random seed on the slope distribution')

if save_figure:
    plt.savefig('slope_seed_dist_{}_208.png'.format(dataset_plot))




