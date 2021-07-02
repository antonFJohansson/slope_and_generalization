

from networks import *

import os
import torch
import torch.nn as nn


class trainingProcedure():
    
    ## Class to perform the training of the networks
    
    def __init__(self, info_dict, train_loader, test_loader, geometric_measure):
        self.num_epochs = info_dict['num_epochs']
        self.mode = info_dict['mode'] 
        
        self.inp_dim = info_dict['inp_dim'] 
        self.out_dim = info_dict['out_dim']
        self.train_loader = train_loader 
        self.test_loader = test_loader
        self.geometric_measure = geometric_measure
        self.batch_size = info_dict['batch_size']
        self.dataset = info_dict['datas']
        self.use_conv = info_dict['conv_model']
        
        self.depth = info_dict['depth']
        self.width = info_dict['width']
        self.lr = info_dict['lr']
        
        self.folder_name = info_dict['folder_name']
        
        self.load_model()
        
    def load_model(self):
        
        
        if not self.use_conv:
            self.my_net = Network(self.inp_dim, self.out_dim, self.depth, self.width, 'relu', 0.0, False)
        else:
            ## Width is the number of channels here
            self.my_net = Network_conv(self.inp_dim, self.out_dim, self.depth, self.width) 

    def train(self):

        min_val_loss = 1000000
        all_train_loss_store = []
        all_test_loss_store = []
        all_train_acc_store = []
        all_test_acc_store = []
        
        opt = torch.optim.SGD(self.my_net.parameters(), lr = self.lr, momentum = 0.8)
        
        if self.mode == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()
            
        ## Measure slope here here
        self.geometric_measure.calculate_geom_properties(self.my_net, -1)
        
        for epoch in range(self.num_epochs):
        
            ## Needed since gradient is shut off for parameters
                ## when measuring the slope
            for param in self.my_net.parameters():
              param.requires_grad = True
            
            
            tot_loss = 0.
            train_acc = 0.
            self.my_net.train()
            for x,y in self.train_loader:
                
                self.my_net.zero_grad()
                out,_ = self.my_net(x)
                loss = loss_fn(out, y)
                
                loss.backward()
                opt.step()
                tot_loss = tot_loss + loss.item()
                out_class = torch.argmax(out,dim=1)
                
                if self.mode == 'classification':
                    train_acc = train_acc + (out_class == y).sum()
                else:
                    train_acc = 0.0
                
            
            tot_loss_v = 0.
            test_acc = 0.
            self.my_net.eval()
            for x,y in self.test_loader: 
            
                
                out,_ = self.my_net(x)
                loss_v = loss_fn(out, y)
                tot_loss_v = tot_loss_v + loss_v.item()
                out_class = torch.argmax(out,dim=1)
                if self.mode == 'classification':
                    test_acc = test_acc + (out_class == y).sum()
                else:
                    test_acc = 0.0
                
            tot_loss = float(tot_loss) / float(len(self.train_loader.dataset))
            tot_loss_v = float(tot_loss_v) / float(len(self.test_loader.dataset))
            train_acc = float(train_acc) / float(len(self.train_loader.dataset))
            test_acc = float(test_acc) / float(len(self.test_loader.dataset))
                
            print('Epoch {} Train Loss {:.4f} Test loss {:.4f} Train acc {:.4f} Test acc {:.4f}'.format(epoch, tot_loss, tot_loss_v, train_acc, test_acc))
            all_train_acc_store.append(train_acc)
            all_test_acc_store.append(test_acc)
            all_train_loss_store.append(tot_loss)
            all_test_loss_store.append(tot_loss_v)
            
            if tot_loss_v < min_val_loss:
                torch.save(self.my_net.state_dict(), os.path.join(self.folder_name, 'my_model.pt'))
                min_val_loss = tot_loss_v
                
            ## Measure slope here
            self.geometric_measure.calculate_geom_properties(self.my_net, epoch)
            
        return all_train_acc_store, all_test_acc_store, all_train_loss_store, all_test_loss_store, self.my_net
            
            