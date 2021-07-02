

## This code obtains the results for the local variations in the slope
## and stores the info in a txt-file which can later be used to plot the
## results

## Put the optimal model for the 3 layer network with 200 neurons in each layer
## in the same folder as this Python-file.

import torch
import torchvision
from resizedDataloader import obtain_resized_data
import numpy as np
from networks import Network

def get_W(my_net, data_pt):

    ## Obtain the weight matrix W in the affine transformation f(x) = Wx + b
        ## Args:
            ## my_net: The network f(x)
            ## data_pt: A data point where the affine transformation should be obtained

        
          for param in my_net.parameters():
              param.requires_grad = False

          out, act_reg = my_net(data_pt, get_list = True)
        
          all_bool_matr = []
        
          for reg in act_reg:
            all_bool_matr.append(np.diag(reg[0].numpy()))
        
        
          all_weights_matr = []
          for n,p in my_net.named_parameters():
            if 'weight' in n:
              all_weights_matr.append(p.numpy())
        
        
          all_bool_matr = all_bool_matr[::-1]
          all_weights_matr = all_weights_matr[::-1]
        
          
          for k in range(len(all_bool_matr)):
            if k == 0:
              W = np.matmul(all_weights_matr[k], all_bool_matr[k])
            else:
              n_W = np.matmul(all_weights_matr[k], all_bool_matr[k])
              W = np.matmul(W, n_W)
            
          W = np.matmul(W, all_weights_matr[-1])
          W = np.transpose(W)
          return W

def get_sing(jac):

            eig = np.linalg.eig(np.matmul(np.transpose(jac), jac))[0]
            
            max_sing = np.sqrt(np.abs(np.max(eig)))
            return max_sing

import argparse
if __name__ == '__main__':

    ds = 'FMNIST'
    img_resize = 28
    train_loader1, _, num_inp, num_out, mode = obtain_resized_data(ds, img_resize, 1)
    
    inp_dim = 784
    out_dim = 10
    width = 200
    depth = 3
    act_func = 'relu'
    drop_out_rate = 0
    use_bn = False
    my_net = Network(inp_dim, out_dim, depth, width, act_func, drop_out_rate, use_bn)
    
    load_model = True
    
    if load_model:
        my_net.load_state_dict(torch.load('my_model.pt'))
    
    num_steps = 150
    num_samples = 500
    num_curves = 250
    
    
    import numpy as np
    rad_list = np.linspace(0.0001, 1.5, num_steps).tolist()
    
    with open('saved_info.txt', 'w') as f:
      f.write('id;list\n')
    
    for c,(x,y) in enumerate(train_loader1):
      
    ## Obtain slope at center of sphere
      W = get_W(my_net, x)
      sing_val1 = get_sing(W)
      if c > num_curves:
        break
      
      diff_plot = []
      dist_plot = []
      
      ## Measure the local slope difference
      for r in rad_list:
    
        sing_diff_store = np.zeros(num_samples)
        for sample in range(num_samples):
          y = torch.clone(x).detach()
          rand_vec = torch.Tensor(1,1,28,28).normal_()
          rand_vec = rand_vec / torch.sqrt(torch.sum(rand_vec**2))
          y = y + r*rand_vec
          W = get_W(my_net, y)
          sing_val2 = get_sing(W)
          #print(sing_val1, sing_val2)
    
          sing_diff = np.abs(sing_val1 - sing_val2)/sing_val1
          sing_diff_store[sample] = sing_diff
    
        diff_plot.append(np.mean(sing_diff_store))
        dist_plot.append(r)
    
    
      with open('saved_info.txt', 'r') as f:
        txt = f.read()
      txt = txt + str(c) + ';' + str(diff_plot) + '\n'
      with open('saved_info.txt', 'w') as f:
        f.write(txt)
    
    
      
    
