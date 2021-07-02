
import torch
import torch.nn as nn
import numpy as np 
import os

from geometric import geometricMeasures
from training import trainingProcedure

from dataloader import load_data
from utils import yield_sub_params, geometricDataLoader

import argparse
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=int)
    
    opt = parser.parse_args()
    
    param_id = opt.param
    param_file = os.path.join('allParams', 'param{}.txt'.format(param_id))
    
    ## Loop over the parameters in the param_file
    for param_info in yield_sub_params(param_file):
        
        
        num_trained_models = param_info[0]
        datas = param_info[1]
        
        batch_size = param_info[2]
        num_estimates = param_info[3]
        measure_every = param_info[4]
        num_epochs = param_info[5]
        base_folder_name = param_info[6] #'Experiment {}'.format(time.time())#param_info[6]
        
        conv_model = param_info[7]
        depth = param_info[8]
        width = param_info[9]
        lr = param_info[10]
        
            
        train_loader_g, test_loader_g, inp_dim, out_dim, mode = load_data(datas, 1, shuffle=False)
        
        ## To only measure the slope at a reduced number of random points
        geom_info = geometricDataLoader(train_loader_g, num_estimates)
        geom_loader = torch.utils.data.DataLoader(geom_info)
        
        info_dict = {'num_trained_models': num_trained_models,
                     'datas': datas,
                     'conv_model': conv_model,
                     'batch_size': batch_size,
                     'num_estimates': num_estimates,
                     'measure_every': measure_every,
                     'num_epochs': num_epochs,
                     'depth': depth,
                     'width': width,
                     'lr': lr,
                     'inp_dim': inp_dim,
                     'out_dim': out_dim,
                     'mode': mode}
                        
        for iii in range(num_trained_models):
            
            train_loader, test_loader, inp_dim, out_dim, mode = load_data(datas, batch_size, shuffle=True)
            
            folder_name = base_folder_name + '{}_{}_{}_{}_{}'.format(datas,conv_model, depth, width, iii)
            info_dict['folder_name'] = folder_name
    
            my_geom = geometricMeasures(info_dict, geom_loader)
            my_training = trainingProcedure(info_dict, train_loader, test_loader, my_geom)
            
            all_train_acc_store, all_test_acc_store, all_train_loss_store, all_test_loss_store, my_net = my_training.train()
                            
                            
            
            ## Create a string which contains all relevant information
            info_str = """
            datasetstart:{}:datasetend
            numepochsstart:{}:numepochsend
            trainaccstart:{}:trainaccend
            testaccstart:{}:testaccend
            trainlossstart:{}:trainlossend
            testlossstart:{}:testlossend
            batchsizestart:{}:batchsizeend
            numestimatesstart:{}:numestimatesend
            fulldictstart:{}:fulldictend
            
            No special info otherwise.
            """.format(datas, num_epochs, all_train_acc_store,
                        all_test_acc_store, all_train_loss_store, all_test_loss_store,
                       batch_size, num_estimates, str(info_dict))
            
            with open(os.path.join(folder_name, 'info.txt'), 'w') as f:
              f.write(info_str)
              
        
        
        
