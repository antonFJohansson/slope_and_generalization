import random
import copy
import torch

import ast
def yield_sub_params(file_name):
    
    with open(file_name,'r') as f:
        txt = f.read()
        f = ast.literal_eval(txt)
    for it in f:
        yield it


class geometricDataLoader():

  ## Creates a special dataloader with the points
    ## where we will calculate the slope    

  def __init__(self, data_loader, num_pts):
    

    self.all_ind = [k for k in range(len(data_loader.dataset))]
    random.shuffle(self.all_ind)
    
    self.geom_ind = self.all_ind[:num_pts]
    
    self.eval_pts = []
    for idx,(x,y) in enumerate(data_loader):
        if idx in self.geom_ind:
            x = torch.squeeze(x, dim = 0)
            self.eval_pts.append((x,y[0]))
        
  def __getitem__(self, idx):
    return self.eval_pts[idx]

  def __len__(self):
    return len(self.eval_pts)










