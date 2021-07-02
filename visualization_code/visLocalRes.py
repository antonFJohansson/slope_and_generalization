

## Code to obtain Figure 6 in the article

import os
import pandas as pd
import ast
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

rad_list = np.linspace(0.0001, 1.5, 150).tolist()

df1 = pd.read_csv('saved_info.txt', sep=';')

figure, ax = plt.subplots(1,1, figsize=(20,8))

mean_arr = np.zeros((df1.shape[0], len(rad_list)))

for k in range(df1.shape[0]):
    
    curr_list = ast.literal_eval(df1['list'][k])
    
    ax.plot(rad_list, curr_list, 'b--', alpha = 0.1)
    mean_arr[k,:] = curr_list

mean_arr = np.mean(mean_arr, axis = 0)
ax.plot(rad_list, mean_arr, 'k', label = 'Mean')

ax.legend()
ax.set_xlabel('Distance')
ax.set_ylabel('Relative slope difference')
ax.grid(axis='y')
ax.set_title('Local variations in slope')


plt.savefig('relative_local_slope_208.png')



