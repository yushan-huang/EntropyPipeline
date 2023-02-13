import pandas as pd
import numpy as np
from torch.utils.data import Dataset

dataset_entropy = pd.read_csv('dataset_all_entropy.csv', usecols=['phase_entropy','dispersion_entropy','approximate_entropy','fuzzy_entropy','label'])
dataset_entropy = np.array(dataset_entropy)
inputs2 = dataset_entropy[:,:-1]
labels2 = dataset_entropy[:,-1:].astype('float32')

#normalization
min_=[]
max_=[]
for i in range(inputs2.shape[1]):
    min_.append(inputs2[:,i].min())
    max_.append(inputs2[:,i].max())
min_ = np.array(min_)
max_ = np.array(max_)
inputs2 = (inputs2-min_)/(max_ - min_)

#unsquezze
inputs2 = inputs2.reshape([-1,1,4]).astype('float32')


class DS(Dataset):
  def __init__(self, inputs,labels):
      self.inputs=inputs
      self.labels=labels
      
  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx],self.labels[idx]


    

    
    