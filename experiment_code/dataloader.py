



import pickle
import torch
import os
import torchvision
import numpy as np

def load_data(datas, batch_size, shuffle):

    ## Loads the data
    ## Args:
        ## datas: The dataset, MNIST, KMNIST, FMNIST, Forest Cover
        ## batch_size: batch size for data loader
        ## shuffle: Randomize the data loader or not
    ## Returns:
        ## train_loader: Data loader with training data
        ## val_loader: Data loader with validation data
        ## num_inp: The number of features in the data set
        ## num_out: Number of classes to predict in the data set
        ## mode: Classification or regression
    
    if datas == 'MNIST':
      
      ## Some issue with loading MNIST for PyTorch so use tensorflow instead
      import tensorflow as tf
      mnist = tf.keras.datasets.mnist
      (x_train, y_train), (x_test, y_test) = mnist.load_data() 

      x_train = x_train.astype('float32') 
      x_test = x_test.astype('float32')

      x_train /= 255 #rescaling the grayscale value from 0 to 255 to a float between 0 and 1 for both train and test
      x_test /= 255

      ## Normalization
      x_train = (x_train - 0.1307)/0.3081
      x_test = (x_test - 0.1307)/0.3081

      x_train = torch.from_numpy(x_train).float()
      x_test = torch.from_numpy(x_test).float()

      y_train = torch.from_numpy(y_train).long()
      y_test = torch.from_numpy(y_test).long()
      full_train = [(torch.unsqueeze(x_train[i,:], dim = 0), y_train[i]) for i in range(x_train.shape[0])]
      full_test = [(torch.unsqueeze(x_test[i,:], dim=0), y_test[i]) for i in range(x_test.shape[0])]

      if shuffle == True:
          train_loader = torch.utils.data.DataLoader(full_train, batch_size=batch_size, shuffle= True)  
      else:
          train_loader = torch.utils.data.DataLoader(full_train, batch_size=batch_size)  
      
      val_loader = torch.utils.data.DataLoader(full_test, batch_size=batch_size)
      num_inp = 784
      num_out = 10
      mode = 'classification'


    elif datas == 'FMNIST':
      transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.2860,), (0.3530,))])
      train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformImg)
      
      test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformImg)  
      
      if shuffle == True:
          train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle= True)  
      else:
          train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)  
      
      val_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
      num_inp = 784
      num_out = 10
      mode = 'classification'

    elif datas == 'KMNIST':
      transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1918,), (0.3483,))])
      train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transformImg)
      
      test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transformImg)  
      
      if shuffle == True:
          train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle= True)  
      else:
          train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
      
      val_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
      
      num_inp = 784
      num_out = 10
      mode = 'classification'

    elif datas == 'CIFAR':
      transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4915, 0.4821, 0.4462), (0.2472, 0.2437, 0.2617))])
      train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformImg)
      test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformImg)
      if shuffle == True:
          train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle= True)  
      else:
          train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
      val_loader = torch.utils.data.DataLoader(test, batch_size = batch_size)
      num_inp = 3*32*32
      num_out = 10
      mode = 'classification'
      
    elif datas == 'fc':
        npzfile = np.load(os.path.join('other_data', 'forest_cover_saved.npz'))
        num_inp = 54
        num_out = 7
        mode = 'classification'
        
        train_x = npzfile['train_x']
        val_x = npzfile['val_x']
        # test_train_x = npzfile['test_train_x']
        # test_full_x = npzfile['test_full_x']
        # full_x = npzfile['full_x']
        
        train_y = npzfile['train_y']
        val_y = npzfile['val_y']
        # full_y = npzfile['full_y']
        # test_y = npzfile['test_y']
        
        
        
        train_data = [(torch.from_numpy(train_x[i,:]).float(), torch.tensor(train_y[i])) for i in range(train_x.shape[0])]
        val_data = [(torch.from_numpy(val_x[i,:]).float(),torch.tensor(val_y[i])) for i in range(val_x.shape[0])]
        
        if shuffle == True:
            train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size = batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size)
        
    return train_loader, val_loader, num_inp, num_out, mode