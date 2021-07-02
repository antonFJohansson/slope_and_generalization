

import torchvision
import torch 

## When we resize the data then we also need new normalization values

kmnist_dict = {28: {'mean': 0.19176215085399395, 'std': 0.3483428318598421},
               42: {'mean': 0.19201724195646142, 'std': 0.32495374649111314},
               56: {'mean': 0.19213380421748777, 'std': 0.32356474711688504},
               70: {'mean': 0.19191531087903912, 'std': 0.32567760378002775},
               84: {'mean': 0.1917621516521105, 'std': 0.3283886542120846},
               98: {'mean': 0.19187130652484208, 'std': 0.3258502460858443}}

fmnist_dict = {28: {'mean': 0.28604060345085297, 'std': 0.35302424986205894},
               42: {'mean': 0.28641480256164964, 'std': 0.338113739280157},
               56: {'mean': 0.2865808423100686, 'std': 0.33723361091309967},
               70: {'mean': 0.28626510958758505, 'std': 0.33849135034057287},
               84: {'mean': 0.286040602966447, 'std': 0.3401812147772854},
               98: {'mean': 0.2862005807313359, 'std': 0.33859574645127033}}

mnist_dict = {28: {'mean': 0.1306604784907127, 'std': 0.30810780376273744},
              42: {'mean': 0.13080086766803076, 'std': 0.2882446837111215},
              56: {'mean': 0.13085866431411433, 'std': 0.2870681753743813},
              70: {'mean': 0.1307445361660289, 'std': 0.28888007898524243},
              84: {'mean': 0.1306604791135204, 'std': 0.2911716020074056},
              98: {'mean': 0.13072051983819677, 'std': 0.2890167962278409}}

def obtain_resized_data(ds, img_resize, batch_size):

  if ds == 'MNIST':

    m = mnist_dict[img_resize]['mean']
    s = mnist_dict[img_resize]['std']


    transformImg = torchvision.transforms.Compose([torchvision.transforms.Resize(img_resize), torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((m,), (s,))])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)  
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle= True)  
    test_loader_large = torch.utils.data.DataLoader(test, batch_size=batch_size)  
    test_loader = torch.utils.data.DataLoader(test)
  elif ds == 'KMNIST':

    m = kmnist_dict[img_resize]['mean']
    s = kmnist_dict[img_resize]['std']

    transformImg = torchvision.transforms.Compose([torchvision.transforms.Resize(img_resize), torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((m,), (s,))])
    train = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transformImg)
    
    test = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transformImg)  
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle= True)  
    test_loader_large = torch.utils.data.DataLoader(test, batch_size=batch_size)  
    test_loader = torch.utils.data.DataLoader(test)
  elif ds == 'FMNIST':
    
    m = fmnist_dict[img_resize]['mean']
    s = fmnist_dict[img_resize]['std']

    transformImg = torchvision.transforms.Compose([torchvision.transforms.Resize(img_resize), torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((m,), (s,))])
    train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformImg)
    
    test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformImg)  
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle = True)  
    test_loader_large = torch.utils.data.DataLoader(test, batch_size=batch_size)  
    test_loader = torch.utils.data.DataLoader(test)

  num_inp = img_resize*img_resize 
  num_out = 10
  mode = 'classification'

  return train_loader, test_loader, num_inp, num_out, mode





