import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from utils.augmentations import augmentation
import torch

class UCR_Dataset(Dataset):
    def __init__(self, args, flag='train') -> None:
        super().__init__()
        self.subdataset = args.subdataset
        self.seq_len=args.seq_len
        self.pred_len=args.pred_len
        self.flag=flag
        self.__read_data__()

    def __read_data__(self):

        assert int(self.subdataset)<250 # there are 250 subdatasets
        base_path='./dataset/UCR_Anomaly_FullData'
        subdataset_list=os.listdir(base_path)
        print("Loading {} of UCR dataset".format(subdataset_list[self.subdataset]))
        with open(os.path.join(base_path,subdataset_list[self.subdataset])) as f:
            print(subdataset_list[self.subdataset])
            b=subdataset_list[self.subdataset].split('_')
            train_end=int(b[4])
            AD_start=int(b[5])
            AD_end=int(b[6][:-4])
            print(train_end,AD_start,AD_end)
            a=f.readlines()
            for i in range(len(a)):
                a[i]=float(a[i].strip())
            #print(a)
            a=torch.FloatTensor(a)

            a=(a-a.median())/a.std()
            
            self.train=torch.FloatTensor(a[:train_end]).unsqueeze(dim=-1)
            self.test=torch.FloatTensor(a[train_end:])
            self.label=torch.zeros(self.test.shape)
            self.label[AD_start-train_end:AD_end-train_end]=1
            self.test=self.test.unsqueeze(dim=-1)
            #print(label.sum(),label.shape,label[AD_start-train_end])
            print(len(self.train),len(self.test))

    def __len__(self) -> int:
        if self.flag=='train':
            return len(self.train)-self.seq_len-self.pred_len
        else:
            return len(self.test)-self.pred_len-self.seq_len

    def __getitem__(self, idx):
        if self.flag=='train':
            x=self.train[idx:idx+self.seq_len]
            y=self.train[idx+self.seq_len:idx+self.seq_len+self.pred_len]
            return x,y
        else:
            x=self.test[idx:idx+self.seq_len]
            y=self.test[idx+self.seq_len:idx+self.seq_len+self.pred_len]
            # print(y.shape)
            return x,y,self.label[idx+self.seq_len:idx+self.seq_len+self.pred_len]



'''
def load_UCRdataset(dataset, subdataset, use_dim="all", root_dir="../", nrows=None):
    """
    use_dim: dimension used in multivariate timeseries
    """
    assert int(subdataset)<250 # there are 250 subdatasets
    base_path='./data/UCR_Anomaly_FullData'
    subdataset_list=os.listdir(base_path)
    print("Loading {} of {} dataset".format(subdataset_list[subdataset], dataset))
    x_dim = 1
    # path = data_path_dict[dataset]

    if dataset=='UCR':
        with open(os.path.join(base_path,subdataset_list[subdataset])) as f:
            print(subdataset_list[subdataset])
            b=subdataset_list[subdataset].split('_')
            train_end=int(b[4])
            AD_start=int(b[5])
            AD_end=int(b[6][:-4])
            print(train_end,AD_start,AD_end)

            a=f.readlines()
            for i in range(len(a)):
                a[i]=float(a[i].strip())
            #print(a)
            a=torch.FloatTensor(a)

            a=(a-a.median())/a.std()
            
            train=torch.FloatTensor(a[:train_end]).unsqueeze(dim=-1)
            test=torch.FloatTensor(a[train_end:])
            label=torch.zeros(test.shape)
            label[AD_start-train_end:AD_end-train_end]=1
            test=test.unsqueeze(dim=-1)
            #print(label.sum(),label.shape,label[AD_start-train_end])
            print(len(train),len(test))


            return {'dim':1,'test_labels':label,'train':train,'test':test,'train_end':train_end,'AD_start':AD_start,'AD_end':AD_end}

'''