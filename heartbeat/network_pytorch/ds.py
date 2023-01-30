import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# with open('/Users/yh621/Desktop/dcarte/heartbeat/dataset_all_baseline.csv') as f:
#     head = f.readline().strip().split(',')
#     head[0]='index'
#     datas = [[eval(ii) for ii in i.strip().split(',')] for i in f.readlines()]
#     datas = np.array(datas)
#     inputs1 = datas[:,1:-1]
#     labels1 = datas[:,-1:].astype('float32')
    
# with open('/Users/yh621/Desktop/dcarte/heartbeat/dataset_all_entropy.csv') as f:
#     head = f.readline().strip().split(',')
#     head[0]='index'
#     datas = [[eval(ii) for ii in i.strip().split(',')] for i in f.readlines()]
#     datas = np.array(datas)
#     inputs2 = datas[:,1:-1]
#     labels2 = datas[:,-1:].astype('float32')

dataset_baseline = pd.read_csv('/Users/yh621/Desktop/dcarte/heartbeat/dataset_all_baseline.csv', usecols=['gait_avg','gait_cv','label'])
dataset_baseline = np.array(dataset_baseline)
inputs1 = dataset_baseline[:,:-1]
labels1 = dataset_baseline[:,-1:].astype('float32')

dataset_entropy = pd.read_csv('/Users/yh621/Desktop/dcarte/heartbeat/dataset_all_entropy.csv', usecols=['phase_entropy','dispersion_entropy','approximate_entropy','fuzzy_entropy','label'])
dataset_entropy = np.array(dataset_entropy)
inputs2 = dataset_entropy[:,:-1]
labels2 = dataset_entropy[:,-1:].astype('float32')


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
inputs1 = inputs1.reshape([-1,1,2]).astype('float32')
inputs2 = inputs2.reshape([-1,1,4]).astype('float32')


class DS(Dataset):
  def __init__(self, inputs,labels):
      self.inputs=inputs
      self.labels=labels
      
  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx],self.labels[idx]


    

    
    