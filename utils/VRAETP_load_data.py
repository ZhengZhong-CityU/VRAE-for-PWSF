# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:46:19 2019

@author: zhengzhong
"""

'''
Read the wind speed data and split them into training, validation and test dataset

'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

#import sys 
#sys.path.append("..") 

class Read_data():
    def __init__(self,args):
        self.args=args
        #ratio: a percentage to split the data into train and test
        #train_length: the length of train data to train a VRAETP model
        #timesteps: the lenth of data used to make a prediction
        #pred_length:the length of data to be predicted, k
        #shuffle: bool
        #norm: bool
              
        print('{} dataset loading...'.format(self.args.dataset))
 
        self.raw_data_pd=pd.read_csv(args.data_path)[0:self.args.data_length]
        self.raw_data_pd=self.raw_data_pd[self.args.inputs_index]
    
        self.raw_data=self.raw_data_pd.values
        
        print('raw data shape: {}'.format(len(self.raw_data)))
        
        self.data_split() #split the wind speed data to different datasets
        
        # dataset normalization
        if self.args.norm:
            self.normalization()
        
        self.train_np=self.data_to_timeseries(self.train_np, self.args.train_length) #convert the wind speed data to T+k columns
        self.val_np=self.data_to_timeseries(self.val_np, self.args.timesteps+self.args.pred_length)
        self.test_np=self.data_to_timeseries(self.test_np, self.args.timesteps+self.args.pred_length) #convert the wind speed data to T+k columns
        
        #drop the row which contains none
        self.train_np=self.train_np[~np.isnan(self.train_np).any(axis=(1,2))]
        self.val_np=self.val_np[~np.isnan(self.val_np).any(axis=(1,2))]
        self.test_np=self.test_np[~np.isnan(self.test_np).any(axis=(1,2))]
                       
        if self.args.shuffle: #shuffle the data with seed
            self.data_shuffle()
        
        # convert the np dataset to dataloader
        self.np_to_dataloader()
        
        print('data import finished')
        print('dataset statistics: num_train: {} num_val: {} num_test:{} time_step: {} input_dim: {}'
              .format(self.train_np.shape[0], self.val_np.shape[0], self.test_np.shape[0], self.args.timesteps, self.train_np.shape[1]))
             
    def data_to_timeseries(self, data, length):
        time_data=[]
        N=len(data)
        x_length=length
        i=0
        while i+x_length<=N:
            data_item=data[i:i+x_length]
            time_data.append(data_item)
            i+=1
        
        array_data=np.array(time_data)
        if array_data.ndim==2:
            array_data=np.expand_dims(array_data,-1)  ## expand shape of batch, length to batch, 1, length
            
        return array_data.transpose(0,2,1)  # return shape of batch, dim, length
    
    def data_split(self):
        N=self.raw_data.shape[0]
        train_p=self.args.ratio_list[0]/sum(self.args.ratio_list)
        test_p=self.args.ratio_list[-1]/sum(self.args.ratio_list)
        
        train_length=int(N*train_p)
        train_val_length=int(N*(1-test_p))        
    
        self.train_np=self.raw_data[0:train_length]
        self.val_np=self.raw_data[train_length:train_val_length]
        self.test_np=self.raw_data[train_val_length:]
        
    def normalization(self):
        #mean std
        self.mean=np.nanmean(self.train_np,axis=0)
        self.std=np.nanstd(self.train_np,axis=0,ddof=1)
        self.train_np=(self.train_np-self.mean)/self.std
        self.val_np=(self.val_np-self.mean)/self.std
        self.test_np=(self.test_np-self.mean)/self.std
        
        #min max
#        self.min_value=np.min(self.train_np,axis=(0,1))
#        self.max_value=np.max(self.train_np,axis=(0,1))
#        self.train_np=(self.train_np-self.min_value)/(self.max_value-self.min_value)
#        self.val_np=(self.val_np-self.min_value)/(self.max_value-self.min_value)
#        self.test_np=(self.test_np-self.min_value)/(self.max_value-self.min_value)
        
    def data_shuffle(self):
        np.random.seed(101)
        np.random.shuffle(self.train_np)
        
    def np_to_dataloader(self):
        # from numpy to dataset loader
        tensor_train=torch.from_numpy(self.train_np)
        tensor_val=torch.from_numpy(self.val_np)
        tensor_test=torch.from_numpy(self.test_np)
        
        train_dataset=TensorDataset(tensor_train)
        val_dataset=TensorDataset(tensor_val)
        test_dataset=TensorDataset(tensor_test)
        
        self.train_loader=DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
        self.val_loader=DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
        self.test_loader=DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
        
        print('data loader statistics: num_train_batch: {} num_val_batch: {} num_test_batch:{}'
              .format(len(self.train_loader), len(self.val_loader), len(self.test_loader)))

    
def random_sample(dataset, batch_size=64):
    N=dataset.shape[0]
    tensor_df=TensorDataset(torch.from_numpy(dataset))
    idx = np.random.permutation(N)
    train_index=idx[:batch_size]
    samples = tensor_df[train_index]   
    return samples[0]


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Read a dataset and convert it to time series') 
    parser.add_argument('--data_path', type=str, default='../dataset/H1-01F.xlsx', help='relative path to save the csv data')
    parser.add_argument('--ratio_list',nargs='+',type=int, default=[8,1,1])  # training validation and test percentage
    parser.add_argument('--train_length',type=int, default=50)
    parser.add_argument('--timesteps',type=int, default=40)
    parser.add_argument('--pred_length', type=int, default=6, help='10 minutes for one step')
    parser.add_argument('--shuffle',type=bool,default=False)
    parser.add_argument('--norm',type=bool,default=False)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dataset', type=str, default='ChengduPM', help= 'AE or PC or PM or WP')
    parser.add_argument('--data_length',type=int,default=100000) # PC 2075259, #PM  43824  #AE 19735 #WP 20432
    parser.add_argument('--inputs_index',nargs='+', type=str, default=['Appliances','Windspeed'])
    args = parser.parse_args()
    
    
    dataset_list=['BeijingPM', 'ChengduPM', 'ShanghaiPM', 'GuangzhouPM', 'ShenyangPM']
    for dataset in dataset_list:
        
        args.dataset=dataset
    
        if args.dataset=='AE':
            args.data_path='../dataset/energydata_complete.csv' 
            args.inputs_index=['Windspeed']
        
        if 'PM' in args.dataset:
            args.data_path='../dataset/{}20100101_20151231.csv'.format(args.dataset)
            args.inputs_index=['Iws']
             
        dataset=Read_data(args)
    
#    for batch_id, samples in enumerate(dataset.test_loader):
#        test_samples=samples[0].type(torch.FloatTensor)
#        x=test_samples[0:2,:,:args.timesteps]
#        y=test_samples[0:2,:,args.timesteps:].detach().numpy()
#        print(x)
#        print(y)
#        break
        
    
#    a=random_sample(dataset.train_np)
#    b=random_sample(dataset.val_np)
#    c=random_sample(dataset.test_np)
#    print(a.shape)
#    print(b.shape)
#    print(c.shape)
#    
#    print(np.mean(dataset.train_np,(0,2)))
#    print(np.std(dataset.train_np,(0,2)))
#    print(np.mean(dataset.val_np,(0,2)))
#    print(np.mean(dataset.test_np,(0,2)))
        
        
        ###original wind speed 
        import matplotlib.pyplot as plt
        import os
        Fontsize=20
        
        file=os.path.join('Figures', args.dataset)
        if not os.path.exists(file):
            os.makedirs(file)
      
#        path=os.path.join(file,'original wind speed.png')
#        
#        fig=plt.figure(figsize=((16,8)))
#        plt.plot(dataset.train_np[0:200,0,0],'k',marker='*')
#        plt.xlabel('Time (hour)',fontsize=Fontsize)
#        plt.ylabel('Wind speed (m/s)',fontsize=Fontsize)
#        plt.xticks(fontsize=Fontsize)
#        plt.yticks(fontsize=Fontsize)
#        fig.savefig(path, dpi=600, bbox_inches='tight')
        
        ###original wind speed distribution
        import seaborn as sns
        sns.set_style("white")
        fig=plt.figure(figsize=((16,8)))
        path=os.path.join(file,'original wind speed distribution.png')
        sns.kdeplot(dataset.train_np[:,0,0])
        
        
        
    
    

    
#    

        
    
    