
import numpy as np
from torch.utils.data import Dataset

with open('dataset_entropy.csv') as f:
    head = f.readline().strip().split(',')
    head[0]='index'
    datas = [[eval(ii) for ii in i.strip().split(',')] for i in f.readlines()]
    datas = np.array(datas)
    inputs1 = datas[:,1:-1]
    labels1 = datas[:,-1:].astype('float32')
    
with open('dataset_baseline.csv') as f:
    head = f.readline().strip().split(',')
    head[0]='index'
    datas = [[eval(ii) for ii in i.strip().split(',')] for i in f.readlines()]
    datas = np.array(datas)
    inputs2 = datas[:,1:-1]
    labels2 = datas[:,-1:].astype('float32')


MODELFLAG = 'MLP'

#normalisation 1
min_=[]
max_=[]
for i in range(inputs1.shape[1]):
    min_.append(inputs1[:,i].min())
    max_.append(inputs1[:,i].max())
min_ = np.array(min_)
max_ = np.array(max_)
inputs1 = (inputs1-min_)/(max_ - min_)

#normalisation 2
min_=[]
max_=[]
for i in range(inputs2.shape[1]):
    min_.append(inputs2[:,i].min())
    max_.append(inputs2[:,i].max())
min_ = np.array(min_)
max_ = np.array(max_)
inputs2 = (inputs2-min_)/(max_ - min_)


#unsquezze
if MODELFLAG=='LSTM':
    timesteps = 3
    inputs1 = inputs1.reshape([-1,timesteps,12]).astype('float32')
    inputs2 = inputs2.reshape([-1,timesteps,10]).astype('float32')

    labels1 = labels1.reshape([-1,timesteps,1]).astype('float32')
    labels2 = labels2.reshape([-1,timesteps,1]).astype('float32')
elif MODELFLAG=='MLP':
    inputs1 = inputs1.reshape([-1,1,12]).astype('float32')
    inputs2 = inputs2.reshape([-1,1,10]).astype('float32')
else:
    print('Error')


class DS(Dataset):
  def __init__(self, inputs,labels):
      self.inputs=inputs
      self.labels=labels
      
  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx],self.labels[idx]


    

    
    