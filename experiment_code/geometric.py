
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os


class geometricMeasures():
    
    ## Class to measure geometric properties of the functions
    
    def __init__(self, info_dict, geom_loader):
        self.inp_dim = info_dict['inp_dim'] 
        self.out_dim = info_dict['out_dim']
        self.geom_loader = geom_loader
        self.dataset = info_dict['datas']
        self.conv_model = info_dict['conv_model']
        self.measure_every = info_dict['measure_every']
        self.folder_name = info_dict['folder_name']
        self.mode = info_dict['mode']

        ## Create folder where the data should be saved
        os.mkdir(self.folder_name)
        
        
    def get_W(self,my_net, data_pt):
        
        ## Obtain the weight matrix W in the affine transformation f(x) = Wx + b
        ## Args:
            ## my_net: The network f(x)
            ## data_pt: A data point where the affine transformation should be obtained

          out, act_reg = my_net(data_pt, get_list = True)
        
          all_bool_matr = []
          
          ## Create the matrices Z
          for reg in act_reg:
            all_bool_matr.append(np.diag(reg[0].numpy()))
        
        
          all_weights_matr = []
          for n,p in my_net.named_parameters():
            if 'weight' in n:
              all_weights_matr.append(p.numpy())
        
          ## Matrix multiplication should be performed from last to first layer
          all_bool_matr = all_bool_matr[::-1]
          all_weights_matr = all_weights_matr[::-1]
        
          ## Obtain the matrix W
          for k in range(len(all_bool_matr)):
            if k == 0:
              W = np.matmul(all_weights_matr[k], all_bool_matr[k])
            else:
              n_W = np.matmul(all_weights_matr[k], all_bool_matr[k])
              W = np.matmul(W, n_W)
            
          W = np.matmul(W, all_weights_matr[-1])
          W = np.transpose(W)
          return W

    
    def get_jacobian(self, my_net, data_pt):
        
        ## Obtain the jacobian by using PyTorchs grad functions
        ## Args:
            ## my_net: The network f(x)
            ## data_pt: A data point where the affine transformation should be obtained
          
        jac = np.zeros((self.inp_dim, self.out_dim))
        
        #out,_ = my_net(data_pt)

        for kkk in range(self.out_dim):
          
          grad_v = torch.zeros(1,self.out_dim)
          
          grad_v[0, kkk] = 1.
          x = torch.clone(data_pt)
          x.requires_grad = True
          #print(x)
          out,_ = my_net(x)
          out.backward(grad_v)
          jac_v = torch.flatten(x.grad)
          jac[:,kkk] = jac_v.numpy()
        
        return jac
          

    def get_sing(self,jac):
        
        ## Get the largest singular value as the square root of the eigenvalues to X^TX
        ## Args:
            ## jac: Jacobian matrix
         
            eig = np.linalg.eig(np.matmul(np.transpose(jac), jac))[0]
            
            max_sing = np.sqrt(np.abs(np.max(eig)))

            return max_sing

    def obtain_sing_vectors(self,jac):
        
        ## Obtain the singular vectors
        ## Args:
            ## jac: Jacobian matrix

        jac = np.transpose(jac)
        U,SIGMA,V = np.linalg.svd(jac)
        u_vec = U[:,0]
        v_vec = V[:,0]
        return u_vec, v_vec
        
    
  
    def calculate_geom_properties(self, my_net, epoch):

    ## Obtain the geometric properties at the given epochs
    ## In this code, only the slope is measured
    ## Args:
        ## my_net: Network to measure the slope for
        ## epoch: The epoch when the slope is measured
    
      ## If the slope should not be measured every epoch
      if (epoch % self.measure_every) != 0:
          return
          
      file_name = 'e_{}.txt'.format(epoch)

      store_all_info = {}
      
      if self.mode == 'classification':
          loss_fn = nn.CrossEntropyLoss()
      else:
          loss_fn = nn.MSELoss()
          
      ## Gradients should not be active for parameters when calculating the jacobian
      for param in my_net.parameters():
          param.requires_grad = False
      
      for idx,(x,y) in enumerate(self.geom_loader):
      
        out,_ = my_net(x)
        curr_loss = loss_fn(out, y).item()
        curr_class = y.item()
        curr_probs = F.softmax(out, dim = 1).tolist()[0]

        ## Easier to calculate the jacobian for conv nets with PyTorch grad functions
        if self.conv_model:
            curr_jac = self.get_jacobian(my_net, x)
        else:
            curr_jac = self.get_W(my_net, x)


        curr_sing = self.get_sing(curr_jac)
        
        
        #sing_u, sing_v = self.obtain_sing_vectors(curr_jac)
        
        store_all_info[idx] = {}
        store_all_info[idx]['class'] = curr_class 
        store_all_info[idx]['probs'] = curr_probs 
        
        store_all_info[idx]['max_sing'] = curr_sing
        
        #store_all_info[idx]['u_vec'] = sing_u.tolist()
    
        store_all_info[idx]['id'] = idx
        store_all_info[idx]['loss'] = curr_loss
        
        all_keys = list(store_all_info[0].keys())
        

        if idx == 0:  
          txt = ';'.join(all_keys) + '\n'
        else:
          with open(os.path.join(self.folder_name,file_name), 'r') as f:
            txt = f.read()
      
        all_txt = []
        for k in all_keys:
          all_txt.append(str(store_all_info[idx][k]))
        txt = txt + ';'.join(all_txt) + '\n'
      
        
        with open(os.path.join(self.folder_name,file_name), 'w') as f:
          f.write(txt)
                                                                