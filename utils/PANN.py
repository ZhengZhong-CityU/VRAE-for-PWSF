# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:16:57 2020

@author: zhongzheng
"""

import torch.nn as nn
import torch
from tcn import TemporalConvNet
import numpy as np
from VRAETP_load_data import Read_data, random_sample
import os
from evaluations import MAE, RMSE
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sigmoid_TCN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels, kernel_size=2, dropout=0.2, activation='ReLU', nonlinear=True):
        super(Sigmoid_TCN, self).__init__()
        self.tcn=TemporalConvNet(num_inputs, num_channels, kernel_size, dropout, activation)
        self.linear=nn.Linear(num_channels[-1],num_outputs)
        self.nonlinear=nonlinear
        if self.nonlinear:
            self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        x=self.tcn(x)
        x=self.linear(x.transpose(1,2))
        if self.nonlinear:
            x=self.sigmoid(x)      
        return x.transpose(1,2)     #shape batch, num_outputs, L


class PGRU(nn.Module):
    def __init__(self, num_inputs, rnn_hidden, rnn_layers, output_dim, dropout=0.1):
        super(PGRU, self).__init__()
        self.dropout=dropout
        self.rnn_layers=rnn_layers
        self.rnn_hidden=rnn_hidden
        self.num_inputs=num_inputs
        self.output_dim=output_dim
        self.GRU=nn.GRU(self.num_inputs, self.rnn_hidden, self.rnn_layers, batch_first=True)
        self.linear=nn.Linear(self.rnn_hidden, self.output_dim)
    def forward(self, x):
        # x of shape batch, dim, length
        inputs=x.transpose(1,2)
        initial_state=torch.zeros(self.rnn_layers, inputs.shape[0], self.rnn_hidden).to(device)
        raw_outputs,hidden=self.GRU(inputs, initial_state)  #raw_outputs (batch, L, hidden)
#        print(hidden.shape)
        output=self.linear(F.dropout(raw_outputs, self.dropout))  # (batch, L, output_dim)
        return output.transpose(1,2)  # (batch, output_dim, L)

    
class PANN():
    def __init__(self, args, device):
        self.args=args
        self.device=device
        self.dataset=Read_data(self.args)
        self.build()
        self.trainable_num=self.get_parameter_number()    
    
    def build(self):
        assert self.args.model_type in ['GRU', 'TCN']
        if self.args.model_type=='TCN':
            self.network=Sigmoid_TCN(self.args.num_inputs, self.args.num_inputs, self.args.channels, self.args.kernel_size, nonlinear=False)       
        if self.args.model_type=='GRU':
            self.network=PGRU(self.args.num_inputs, self.args.rnn_hidden, self.args.rnn_layers, self.args.num_inputs, self.args.dropout)
        self.network=self.network.to(self.device)    
        self.optimizer=torch.optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
    
    @property
    def model_dir(self):
        return '{}_{}_{}_{}_{}_{}'.format(self.args.model_type,self.args.timesteps,self.args.pred_length,self.args.lr, self.args.iterations, self.args.dropout) 

    def get_parameter_number(self):
        trainable_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print('Number of trainable parameters: {}: {}; '.format(self.args.model_type,trainable_num))
        return trainable_num 
    
    def train(self):
        print('network training start!')
        self.network.train()
        train_losses=[]
        
        for iteration in range(self.args.iterations):
            real_samples=random_sample(self.dataset.train_np,self.args.batch_size).type(torch.FloatTensor)  ## shape of batch, dim, length+1
            real_samples=real_samples.to(self.device)
            x=real_samples[:,:,:self.args.timesteps]  ## shape of batch, dim, timesteps
            y=real_samples[:,:,-1]
            self.optimizer.zero_grad()
            pred_y=self.network(x)[:,:,-1]
            loss=torch.mean(torch.pow(y-pred_y,2))
            loss.backward()
            self.optimizer.step()
            loss_value=loss.data.item()
            train_losses.append(loss_value)
            
            print('training iteration {}|{}: MSE loss: {}'.format(iteration+1, self.args.iterations, loss_value))
        self.save_losses(train_losses)
        return train_losses
    
    def prediction(self,x):
        self.network.eval()
        x=x.to(self.device)
        pred_y=self.network(x)[:,:,-1]
        generated_y=pred_y.cpu().detach().numpy()
        return  generated_y
    
    def pred_eval(self, mode='validation'):
        
        assert mode in ['validation', 'test']
        
        if mode=='validation':
            dataset= self.dataset.val_loader
            path=self.args.validation_path
        else:
            dataset=self.dataset.test_loader
            path=self.args.test_path
        
        print('{} generation on the {} dataset start!'.format(self.args.model_type, mode)) 
          
        save_path=os.path.join(self.args.dataset, self.model_dir, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            generated_y=[]
            real_y=[]
            for batch_id, samples in enumerate(dataset):
                test_samples=samples[0].type(torch.FloatTensor)
                x=test_samples[:,:,:self.args.timesteps]
                y=test_samples[:,:,-1].detach().numpy()
                real_y.append(y)
                pred_y=self.prediction(x)
                generated_y.append(pred_y)
            generated_y=np.concatenate(generated_y,0)
            real_y=np.concatenate(real_y,0) 
            
            real_y=real_y*self.dataset.std+self.dataset.mean
            generated_y=generated_y*self.dataset.std+self.dataset.mean
            
            self.save_generation(real_y, generated_y, save_path)   
        else:
            print('generation already exists, loading generation from {}'.format(save_path))
            real_y_file=os.path.join(save_path, 'real_y.npy')
            generated_y_file=os.path.join(save_path, 'generated_y.npy')
            real_y=np.load(real_y_file)
            generated_y=np.load(generated_y_file)
            print('generation loading finished')
        return real_y, generated_y
    
    def save_generation(self, real_y, generated_y, save_path):
        print('real samples shape {}; generated samples shape: {}'.format(real_y.shape, generated_y.shape))
        real_y_file=os.path.join(save_path, 'real_y.npy')
        np.save(real_y_file, real_y) 
        generated_y_file=os.path.join(save_path, 'generated_y.npy')
        np.save(generated_y_file, generated_y)   
        print('successfully save the generation to {}'.format(save_path))
            
    def save_model(self):
        save_path=os.path.join(self.args.dataset, self.model_dir, self.args.save_model_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path=os.path.join(save_path, 'model.pth')
        torch.save(self.network.state_dict(), model_path)
        print('model saved to {}'.format(save_path))
    
    def load_model(self):
        load_path=os.path.join(self.args.dataset, self.model_dir, self.args.save_model_path)
        if not os.path.exists(load_path):
            print('load path does not exist')
        else:
            load_path=os.path.join(load_path, 'model.pth')
            self.network.load_state_dict(torch.load(load_path,map_location=device))
            print('model loaded from {}'.format(load_path))
    
    def save_losses(self, train_losses):
        save_path=os.path.join(self.args.dataset,self.model_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save losses
        losses_file=os.path.join(save_path,'losses.npy')
        np.save(losses_file, train_losses)
        
        # save figure
        path=os.path.join(save_path,'training_losses.png')
        fig=plt.figure(figsize=(12,8))
        plt.plot(train_losses, 'b', label='Training losses')
        plt.xlabel('iterations')       
        plt.ylabel('Training losses')
        plt.legend()
        fig.savefig(path)
    def save_score(self, score, mode='validation', metric='MAE'):
        assert mode in ['validation', 'test']
        if mode =='validation':
            path=self.args.validation_path
        else:
            path=self.args.test_path
            
        score_file=os.path.join(self.args.dataset, self.model_dir, path, '{}.txt'.format(metric))
        with open (score_file,'w') as f:
            f.write(str(score))
        print('score {} on the {} dataset saved to {}'.format(score, mode, score_file))
        
def train(dataset, args):
    if args.dataset=='AE':
        args.data_path='../dataset/energydata_complete.csv' 
        args.inputs_index=['Windspeed']
        
    if 'PM' in args.dataset:
        args.data_path='../dataset/{}20100101_20151231.csv'.format(args.dataset)
        args.inputs_index=['Iws']
    args.num_inputs=len(args.inputs_index)
    
    pred_length_list=[1] ## ranging from ten minutes to 60 minutes
    dropout_list=[0,0.1]
     
    for dropout, pred_length in itertools.product(dropout_list, pred_length_list):
        args.dropout=dropout
        args.pred_length=pred_length
        args.train_length=args.timesteps+args.pred_length
    
        ###training stage####    
        model=PANN(args, device)    
        load_path=os.path.join(model.args.dataset, model.model_dir, model.args.save_model_path)
        if not os.path.exists(load_path):
            print('load path does not exist, training from scratch')
            train_losses=model.train()
            model.save_model()
        else:
            print('load path exists, loading...')
            model.load_model()
             
        ### validation stage###
        real_y, final_generation=model.pred_eval('validation')  
        
#        ## if not denormalized, do this part
#        real_y=real_y*model.dataset.std+model.dataset.mean
#        final_generation=final_generation*model.dataset.std+model.dataset.mean
#        path=model.args.validation_path
#        save_path=os.path.join(model.args.dataset, model.model_dir, path)
#        model.save_generation(real_y, final_generation, save_path)
        
        
        MAE_score=MAE(real_y, final_generation, average=args.average)
        model.save_score(MAE_score, mode='validation', metric='MAE')
        RMSE_score=RMSE(real_y, final_generation, average=args.average)
        model.save_score(RMSE_score, mode='validation', metric='RMSE')
        
        ### test stage###
        real_y, final_generation=model.pred_eval('test')
        
#        ## if not denormalized, do this part
#        real_y=real_y*model.dataset.std+model.dataset.mean
#        final_generation=final_generation*model.dataset.std+model.dataset.mean
#        path=model.args.test_path
#        save_path=os.path.join(model.args.dataset, model.model_dir, path)
#        model.save_generation(real_y, final_generation, save_path)
        
        MAE_score=MAE(real_y, final_generation, average=args.average)
        model.save_score(MAE_score, mode='test', metric='MAE')
        RMSE_score=RMSE(real_y, final_generation, average=args.average)
        model.save_score(RMSE_score, mode='test', metric='RMSE')
    
       
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Predictive ANN for multivariate time series generation')
    parser.add_argument('--model_type', choices=['TCN','CNN','GRU','MLP'], default='GRU')
    parser.add_argument('--ratio_list',nargs='+',type=int, default=[8,1,1])
    parser.add_argument('--train_length',type=int, default=50)
    parser.add_argument('--timesteps',type=int, default=40)
    parser.add_argument('--pred_length', type=int, default=6, help='10 minutes for one step')
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--norm',type=bool,default=True)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dataset', type=str, default='GOOG', help= 'AE or PC or PM or WP')
    parser.add_argument('--data_path', type=str, default='../dataset/H1-01F.xlsx', help='relative path to save the csv data')
    parser.add_argument('--data_length',type=int,default=100000) # PC 2075259, #PM  43824  #AE 19735 #WP 20432 
    
    parser.add_argument('--inputs_index',nargs='+', type=str, default=['Appliances','Windspeed'])
    parser.add_argument('--num_inputs', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help='kernel size to use')  
    parser.add_argument('--channels', nargs='+', type=int, default=[32]*5)  ## rf(k=3,n=4)=61
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="adam: learning rate, default=0.0001")
    parser.add_argument("--b1", type=float, default=0.9, 
                        help="adam: decay of first order momentum of gradient, default=0.5")
    parser.add_argument("--b2", type=float, default=0.999, 
                        help="adam: decay of first order momentum of gradient, default=0.999")
    parser.add_argument("--iterations", type=int, default=20000, 
                        help="number of iterations of training, default=5w") 
    parser.add_argument('--save_model_path', type=str, default='saved_models', 
                        help='save model path')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')   
    parser.add_argument('--validation_path', type=str, default='validation_generation', 
                        help='path to save the generation on the validation set')
    parser.add_argument('--test_path', type=str, default='test_generation', 
                        help='path to save the generation on the test set') 
    
    parser.add_argument("--rnn_hidden", type=int, default=32, 
                        help="number of hidden units in the GRU")     
    parser.add_argument("--rnn_layers", type=int, default=4, 
                        help="number of hidden layers in the GRU")  
    parser.add_argument("--dropout", type=float, default=0.0, 
                        help="output dropout in the GRU")  

    parser.add_argument('--average', type=bool, default=True,
                        help='whether to average the CRPS or MAE score')    
    
    args=parser.parse_args() 
    
    
    #specify the seed and device
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    dataset_list=['BeijingPM', 'ChengduPM', 'ShanghaiPM', 'GuangzhouPM', 'ShenyangPM']
    
    for dataset in dataset_list:
        args.dataset=dataset
        train(dataset, args)

    


    

        
    
