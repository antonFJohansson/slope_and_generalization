
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    
    ## Fully connected network
    
    def __init__(self, inp_dim, out_dim, depth, width, act_func, drop_out_rate, use_bn):
        super().__init__()
        
        self.moduleList = []
        fc = nn.Linear(inp_dim, width)
        self.moduleList.append(fc)
        for i in range(depth -  1):
            fc = nn.Linear(width, width)
            self.moduleList.append(fc)
                
        fc = nn.Linear(width, out_dim)
        self.moduleList.append(fc)
        self.moduleList = nn.ModuleList(self.moduleList)
        
        self.inp_dim = inp_dim

        self.depth = depth
        self.width = width
        
        self.bn_list = []
        self.use_bn = use_bn
        if use_bn:
            for i in range(self.depth):
                bn = nn.BatchNorm1d(width)
                self.bn_list.append(bn)
            self.bn_moduleList = nn.ModuleList(self.bn_list)
        
        act_dict = {'relu': [F.relu, 0],
                    'tanh': [F.tanh, 0],
                    'sigmoid': [F.sigmoid, 0.5]}
        
        self.act_func = act_dict[act_func][0]
        self.thres = act_dict[act_func][1]
        self.out_dim = out_dim

        if not drop_out_rate is None:
          self.drop = nn.Dropout(drop_out_rate)
          self.use_drop = True
        else:
          self.use_drop = False

        
    def forward(self, x, get_list = False):

        ## If it is an image then we flatten it
        if len(x.shape) > 2:
              x = x.view(-1, 784)

        
        prev_output = []
        act_list = []
        for idx,fc in enumerate(self.moduleList):
            

            if idx == (len(self.moduleList) - 1):
                ## Output layer
                x = fc(x)
            else:
                prev_output.append(fc(x))

                x = fc(x)
                if self.use_drop:
                  x = self.drop(x)
                x = self.act_func(x)
                #x = self.act_func(fc(x))
                act_list.append(x>self.thres)
                if self.use_bn:
                    x = self.bn_moduleList[idx](x)
                
        if not get_list:
          return x, torch.cat(act_list, dim=1).tolist()
        else:
          return x, act_list


class Network_conv(nn.Module):
    
    ## Convolutional network
    
    def __init__(self, inp_dim, out_dim, num_layers, num_channels):
        super().__init__()
        
        moduleList = []

        cn = nn.Conv2d(1, num_channels, kernel_size=(5, 5), padding=2)
        moduleList.append(cn)
        for i in range(num_layers - 1):
              cn = nn.Conv2d(num_channels, num_channels, kernel_size=(5, 5), padding=2)
              moduleList.append(cn)
        
        self.final_dim = num_channels*28*28
        self.fc = nn.Linear(self.final_dim, 10)    
        self.moduleList = nn.ModuleList(moduleList)

        
    def forward(self, x):
        
        for cn in self.moduleList:
          x = F.relu(cn(x))

        x = x.view(-1, self.final_dim)
        x = self.fc(x)
        
        ## The second return value is just due to legacy reasons
        return x, 0
