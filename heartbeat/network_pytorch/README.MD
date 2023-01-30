# 相关代码文件
ds.py 数据读取及normalization的代码
net.py 网络搭建的代码
lr.py  warmup学习策略实现代码

train_ds1.py 训练数据集dataset_1.csv
train_ds2.py 训练数据集dataset_2.csv



#一些指标
## 数据集1
epoch=4991, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4992, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4993, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4994, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4995, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4996, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4997, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4998, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
epoch=4999, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649   

## 数据集2
input_size=5,num_layers=2
epoch=4993, train:loss=0.468,recall=0.705,f1=0.479  dev:loss=0.512,recall=0.600,f1=0.416
epoch=4994, train:loss=0.468,recall=0.705,f1=0.479  dev:loss=0.512,recall=0.600,f1=0.416
epoch=4995, train:loss=0.468,recall=0.705,f1=0.479  dev:loss=0.512,recall=0.600,f1=0.416
epoch=4996, train:loss=0.468,recall=0.705,f1=0.479  dev:loss=0.512,recall=0.600,f1=0.416
epoch=4997, train:loss=0.468,recall=0.705,f1=0.479  dev:loss=0.512,recall=0.600,f1=0.416
epoch=4998, train:loss=0.468,recall=0.705,f1=0.479  dev:loss=0.512,recall=0.600,f1=0.416
epoch=4999, train:loss=0.468,recall=0.705,f1=0.479  dev:loss=0.512,recall=0.600,f1=0.416
input_size=5,num_layers=1
epoch=4995, train:loss=0.495,recall=0.765,f1=0.450  dev:loss=0.482,recall=0.719,f1=0.452
epoch=4996, train:loss=0.495,recall=0.765,f1=0.450  dev:loss=0.482,recall=0.719,f1=0.452
epoch=4997, train:loss=0.495,recall=0.765,f1=0.450  dev:loss=0.482,recall=0.719,f1=0.452
epoch=4998, train:loss=0.495,recall=0.765,f1=0.450  dev:loss=0.482,recall=0.719,f1=0.452
epoch=4999, train:loss=0.495,recall=0.765,f1=0.450  dev:loss=0.482,recall=0.719,f1=0.452



