import gc
import torch
import random
import pandas as pd
from torch.utils.data import DataLoader
from ds import inputs2,labels2,DS
from net_entropy import RNN
from lr import WarmupCosineLR

recall_tr = []
recall_ts = []
precision_tr = []
precision_ts = []
F1_tr = []
F1_ts = []

for i in range(1):

    label_ = labels2
    input_ = inputs2

    #balance the labels
    positive = [i for i in range(len(label_)) if label_[i]==1]
    negative = [i for i in range(len(label_)) if label_[i]==0]
    random.shuffle(negative)
    index = positive+negative[:len(positive)]
    random.shuffle(index)

    #split dataset
    test = 0.3
    train = index[:-int(len(index)*test)]
    dev   = index[:int(len(index)*test)]

    train_ds = DS(input_[train],label_[train])
    dev_ds   = DS(input_[dev],label_[dev])

    epochs     = 3000
    warm_epoch = 1000
    batch_size = 258
    LR         = 0.5

    train_dl = DataLoader(train_ds, batch_size=len(train), shuffle=True,num_workers=0)
    val_dl   = DataLoader(dev_ds, batch_size=len(dev), shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = RNN(input_size=4,num_layers=2).to(device)

    #set model
    criterion = torch.nn.BCELoss()                        #loss function
    optimizer = torch.optim.SGD(net.parameters(),lr=LR,momentum=0.9)  #optimiser
    cosine_lr = WarmupCosineLR(optimizer, 0.00001, LR, warm_epoch, epochs, 0.1) #cosine annealing

    #recall f1
    def evaluate(out,labels):
        _, prediction = torch.max(out,1)
        TP,FP,FN,TN=0,0,0,0
        for i in range(out.shape[0]):
            if out[i][0]>0.5:
                if labels[i][0]==1.:
                    TP+=1
                else:
                    FN+=1
            else:
                if labels[i][0]==0.:
                    TN+=1
                else:
                    FP+=1
                    
        recall = TP/(TP+FN+0.1)
        precision = TP/(TP+FP+0.1)
        # f1 = (2*precision*recall)/(precision+recall+1)
        f1 = (2*TP)/(2*TP+FP+FN+0.1)
        return recall,f1,precision
        


    #start training
    train_loss = []
    train_acc  = []
    val_loss   = []

    for epoch in range(epochs):
        gc.collect()
        torch.cuda.empty_cache()
        for batch, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            cosine_lr.step()
            
            if batch%1==0:
                recall,f1,precision = evaluate(out,labels)
                #evaluation
                with torch.no_grad():
                    for data in val_dl:
                        inputs, labels = data[0].to(device), data[1].to(device)
                    dev_out = net(inputs)
                    dev_recall,dev_f1,dev_precision = evaluate(dev_out,labels)
                    dev_loss = criterion(dev_out, labels)
                
                train_loss.append(loss.item())
                val_loss.append(dev_loss.item())
                print("epoch=%d, train:loss=%.3f,recall=%.3f,f1=%.3f,precision=%.3f  dev:loss=%.3f,recall=%.3f,f1=%.3f,precision=%.3f,num=%.1f"%(epoch,loss,recall,f1,precision,dev_loss,dev_recall,dev_f1,dev_precision,i))
        
    F1_tr.append(f1)
    F1_ts.append(dev_f1)
    recall_tr.append(recall)
    recall_ts.append(dev_recall)
    precision_tr.append(precision)
    precision_ts.append(dev_precision)

evaluation_final = pd.DataFrame([])
F1_tr = pd.DataFrame(F1_tr)
F1_ts = pd.DataFrame(F1_ts)
recall_tr = pd.DataFrame(recall_tr)
recall_ts = pd.DataFrame(recall_ts)
precision_tr = pd.DataFrame(precision_tr)
precision_ts = pd.DataFrame(precision_ts)

evaluation_final = pd.concat([evaluation_final, F1_tr], axis=1)
evaluation_final = pd.concat([evaluation_final, F1_ts], axis=1)
evaluation_final = pd.concat([evaluation_final, recall_tr], axis=1)
evaluation_final = pd.concat([evaluation_final, recall_ts], axis=1)
evaluation_final = pd.concat([evaluation_final, precision_tr], axis=1)
evaluation_final = pd.concat([evaluation_final, precision_ts], axis=1)

evaluation_final.columns = ['F1_tr','F1_ts','recall_tr','recall_ts','precision_tr','precision_ts']
# evaluation_final.to_csv('n_evaluation_entropy_lstm.csv')
# evaluation_final.to_csv('kkkkkkkkkkk.csv')

import matplotlib.pyplot as plt

x = list(range(len(train_loss)))
plt.plot(x, train_loss,   label="train_loss")
plt.plot(x, val_loss,     label="val_loss")

#plotting
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss")
plt.ylim(0,1)

plt.legend(loc="lower left")

#save fig
#plt.savefig("ds1_loss.jpg")
plt.show()


#results
    # epoch=4991, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4992, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4993, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4994, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4995, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4996, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4997, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4998, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649
    # epoch=4999, train:loss=0.032,recall=0.981,f1=0.654  dev:loss=0.016,recall=0.962,f1=0.649    
        
        
        
        
        
        
        
        
        
  


    

