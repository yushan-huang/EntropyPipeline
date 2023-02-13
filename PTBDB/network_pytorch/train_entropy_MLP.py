import gc
import torch
import random
from torch.utils.data import DataLoader
from ds import inputs2,labels2,DS
from net_entropy import MLP
from lr import WarmupCosineLR
import pandas as pd

recall_tr = []
recall_ts = []
F1_tr = []
F1_ts = []
acc_tr = []
acc_ts = []

for i in range(30):
    label_ = labels2
    input_ = inputs2

    #balance the labels
    positive = [i for i in range(len(label_)) if label_[i]==1]
    negative = [i for i in range(len(label_)) if label_[i]==0]
    random.shuffle(negative)
    index = positive+negative[:len(positive)]
    random.shuffle(index)

    #split the dataset
    test = 0.3
    train = index[:-int(len(index)*test)]
    dev   = index[:int(len(index)*test)]

    train_ds = DS(input_[train],label_[train])
    dev_ds   = DS(input_[dev],label_[dev])

    epochs     = 2000
    warm_epoch = 500
    batch_size = 256
    LR         = 0.1

    train_dl = DataLoader(train_ds, batch_size=len(train), shuffle=True,num_workers=0)
    val_dl   = DataLoader(dev_ds, batch_size=len(dev), shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MLP(input_size=4).to(device)

    #set model
    criterion = torch.nn.BCELoss()                        #loss function
    optimizer = torch.optim.SGD(net.parameters(),lr=LR,momentum=0.9)  #optimizer
    cosine_lr = WarmupCosineLR(optimizer, 0.00001, LR, warm_epoch, epochs, 0.1) #cosine annealing

    # Get the number of parameters in the model
    total_params = sum(p.numel() for p in net.parameters())

    print("Total number of parameters:", total_params)

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
            inputs, labels = data[0].to(device).reshape((-1,4)), data[1].to(device)
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
                        inputs, labels = data[0].to(device).reshape((-1,4)), data[1].to(device)
                    dev_out = net(inputs)
                    dev_recall,dev_f1,dev_precision = evaluate(dev_out,labels)
                    dev_loss = criterion(dev_out, labels)
                
                train_loss.append(loss.item())
                val_loss.append(dev_loss.item())
                print("epoch=%d, train:loss=%.3f,recall=%.3f,f1=%.3f, acc=%.3f  dev:loss=%.3f,recall=%.3f,f1=%.3f, acc=%.3f, num=%.1f"%(epoch,loss,recall,f1, precision,dev_loss,dev_recall,dev_f1,dev_precision,i))
        
    F1_tr.append(f1)
    F1_ts.append(dev_f1)
    recall_tr.append(recall)
    recall_ts.append(dev_recall)
    acc_tr.append(precision)
    acc_ts.append(dev_precision)

evaluation_final = pd.DataFrame([])
F1_tr = pd.DataFrame(F1_tr)
F1_ts = pd.DataFrame(F1_ts)
recall_tr = pd.DataFrame(recall_tr)
recall_ts = pd.DataFrame(recall_ts)
acc_tr = pd.DataFrame(acc_tr)
acc_ts = pd.DataFrame(acc_ts)
evaluation_final = pd.concat([evaluation_final, recall_ts, F1_ts, acc_ts], axis=1)
evaluation_final.columns = ['test_recall','test_f1','test_acc']
evaluation_final.to_csv('n_evaluation_entropy_MLP.csv')

# # import matplotlib.pyplot as plt

# # x = list(range(len(train_loss)))
# # plt.plot(x, train_loss,   label="train_loss")
# # plt.plot(x, val_loss,     label="val_loss")

# # #plotting
# # plt.xlabel("epoch")
# # plt.ylabel("loss")
# # plt.title("loss")

# # plt.legend(loc="lower left")

# # #save fig
# # #plt.savefig("ds1_loss.jpg")
# # plt.show()

        
        
        
        
        
        
  


    


