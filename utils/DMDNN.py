#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:44:41 2021

@author: zhangyan
"""
'''
Reimplementation of the DMDNN proposed by the paper 
Advanced Deep Learning Approach for Probabilistic Wind Speed Forecasting
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from VRAETP_load_data import Read_data, random_sample
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from evaluations import CRPS, RMSE
import os


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def compute_gradient_penalty(D, x):
    
    y=x.clone().detach().requires_grad_(True)

    D_outputs=torch.sum(D(y))
    
    # Get gradient w.r.t.x
    gradients = autograd.grad(
        outputs=D_outputs,
        inputs=y,
        grad_outputs=None,
        create_graph=True,
        retain_graph=False,
        only_inputs=True,
    )[0]

    return gradients


class Con1dBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(Con1dBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1,  self.relu1, self.dropout1)

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class Conv1dNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dilation_size=1, dropout=0.2):
        super(Conv1dNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [Con1dBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size-1) * dilation_size/2), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)   

class DMDNN(nn.Module):
    def __init__(self, num_Gaussians, num_inputs, num_channels, rnn_hidden, rnn_layers, num_outputs, h_dim, kernel_size, dilation_size, dropout):
        super(DMDNN, self).__init__()
        self.num_Gaussians=num_Gaussians
        self.num_outputs=num_outputs
        self.dropout=dropout
        self.rnn_layers=rnn_layers
        self.rnn_hidden=rnn_hidden
        self.num_inputs=num_inputs
        self.num_channels=num_channels
        
        self.CNN=Conv1dNet(num_inputs, num_channels, kernel_size, dilation_size, dropout) ### return [batch, num_channels[-1], length]
        ## gru inputs should be shape of [batch, length, input_dim]
        self.GRU=nn.GRU(num_channels[-1], rnn_hidden, rnn_layers, batch_first=True)
        
        
        self.Dense=nn.Sequential(nn.Linear(rnn_hidden, h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        
        self.linear=nn.Linear(h_dim, int(num_Gaussians*(2*num_outputs+1)))
        self.softmax=nn.Softmax(dim=1)
    def forward(self, x):
        # x of shape batch, dim, length
        
        CNN_outputs=self.CNN(x)  ##  [batch, num_channels[-1], length]
        initial_state=torch.zeros(self.rnn_layers, CNN_outputs.shape[0], self.rnn_hidden).to(device)
        
        GRU_inputs=CNN_outputs.transpose(1,2)  ##  [batch, length, rnn_hidden]
        GRU_outputs,_=self.GRU(GRU_inputs, initial_state)  #GRU_outputs [batch, L, hidden] 
        Dense_outputs=self.Dense(F.dropout(GRU_outputs, self.dropout)) #Dense_outputs [batch, L, h_dim]
        hidden=self.linear(F.dropout(Dense_outputs, self.dropout))[:,-1,:]  ## hidden [batch, int(num_Gaussians*(2*num_outputs+1))]
        pi=self.softmax(hidden[:, 0:self.num_Gaussians])
        mu=hidden[:,self.num_Gaussians:self.num_Gaussians*(self.num_outputs+1)]
        sigma=torch.exp(hidden[:,self.num_Gaussians*(self.num_outputs+1):])
        return pi, mu.reshape(-1, self.num_Gaussians, self.num_outputs), sigma.reshape(-1, self.num_Gaussians, self.num_outputs)
    
def GMMloss(y, pi, mu, sigma):
    # y: batch, output_dim
    # pi: batch, num_Gaussians
    # mu: batch, num_Gaussians,output_dim
    # sigma: batch, num_Gaussians,output_dim
    N, output_dim=y.shape
    t_y=y.unsqueeze(1)
    t_pi=pi.unsqueeze(-1)
    mixture=torch.distributions.normal.Normal(mu, sigma)
    log_prob=mixture.log_prob(t_y)
#    print(log_prob.shape)
    weighted_logprob=log_prob+torch.log(t_pi)
    log_sum=torch.logsumexp(weighted_logprob, dim=1)
    return -torch.sum(log_sum)
    
def GMM_sample(pi, mu, sigma):
    # pi: batch, num_Gaussians
    # mu: batch, num_Gaussians,output_dim
    # sigma: batch, num_Gaussians,output_dim
    C=pi.shape[0]
    mn=torch.distributions.multinomial.Multinomial(1, probs=pi)
    mask=mn.sample().bool().to(device).unsqueeze(-1)
    sample_mu=mu.masked_select(mask).reshape(C,-1)
    sample_sigma=sigma.masked_select(mask).reshape(C,-1)
    sample=torch.normal(sample_mu, sample_sigma)
    return sample
          
        
class DMDNN_model():
    def __init__(self, args, device):
        self.args=args
        self.device=device
        self.dataset=Read_data(self.args)
        self.build()
        self.trainable_num=self.get_parameter_number()               
        
    def build(self):
        self.network=DMDNN(self.args.num_Gaussians, self.args.num_inputs, self.args.num_channels, self.args.rnn_hidden, self.args.rnn_layers, self.args.output_dim, self.args.h_dim, 
                           self.args.kernel_size, self.args.dilation_size, self.args.dropout)
        self.network=self.network.to(self.device)    
        self.optimizer=torch.optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
    
    @property
    def model_dir(self):
        return 'DMDNN_{}_{}_{}_{}_{}'.format(self.args.timesteps, self.args.pred_length, self.args.lr, self.args.iterations, self.args.dropout) 

    def get_parameter_number(self):
        trainable_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print('Number of trainable parameters: {}; '.format(trainable_num))
        return trainable_num         
        
    def train(self):
        print('network training start!')
        self.network.train()
        train_losses=[]
        
        for iteration in range(self.args.iterations):
            real_samples=random_sample(self.dataset.train_np,self.args.batch_size).type(torch.FloatTensor)  ## shape of batch, dim, length+1
            real_samples=real_samples.to(self.device)
            x=real_samples[:,:,:self.args.timesteps]  ## shape of batch, dim, timesteps
            y=real_samples[:,:,-1] ## shape of batch, output_dim
            self.optimizer.zero_grad()
            pi, mu, sigma=self.network(x)
            loss=GMMloss(y, pi, mu, sigma)
            loss.backward()
            self.optimizer.step()
            loss_value=loss.data.item()
            train_losses.append(loss_value)
            
            print('training iteration {}|{}: NLL loss: {}'.format(iteration+1, self.args.iterations, loss_value))
        self.save_losses(train_losses)
        return train_losses 

    def generation(self, x):
#        print('TCGAN generation on samples start!')
        self.network.eval()
        
        x=x.to(self.device)        
        pi, mu, sigma=self.network(x)
        fake_y=GMM_sample(pi, mu, sigma)  ## batch, dim
        generated_y=fake_y.cpu().detach().numpy()
        return  generated_y 

    def generation_eval(self, num_sampling, mode='validation'):
        ### since we will test on the testset for many times, the num_sampling indicates the total number of trials per experiments
        
        assert mode in ['validation', 'test']
        if mode=='validation':
            dataset= self.dataset.val_loader
        else:
            dataset=self.dataset.test_loader
        
        print('TCGAN generation on the {} dataset start!'.format(mode))  
        generation_per=[]  ## generation of each experiment
        real_y=[]
        for batch_id, samples in enumerate(dataset):
            Time_generation=[]
            test_samples=samples[0].type(torch.FloatTensor)
            x=test_samples[:,:,:self.args.timesteps]
            y=test_samples[:,:,-1].detach().numpy()
            real_y.append(y)
            for time in range(num_sampling):
                print('times {}|{} batches {}|{}'.format(time+1, num_sampling, batch_id+1, len(self.dataset.test_loader)))
                generated_y=self.generation(x)
                Time_generation.append(generated_y)              
            Time_generation=np.asarray(Time_generation)
            generation_per.append(Time_generation)
        generated_y=np.concatenate(generation_per,1)
        real_y=np.concatenate(real_y,0)        
        return real_y, generated_y    
    
    def CRPS_generation(self, num_sampling=100, num_experiments=2, mode='validation'):
        '''
        for CRPS evaluation, we need to conduct a number of experiments, for each experiments, we need to sample num_sampling points
        '''
        
        assert mode in ['validation', 'test']
        if mode =='validation':
            path=self.args.validation_path
        else:
            path=self.args.test_path
        
        save_path=os.path.join(self.args.dataset, self.model_dir, path, '{}_{}'.format(num_sampling, num_experiments))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            generated_y=[]
            for index in range(num_experiments):
                real_y, generation_per=self.generation_eval(num_sampling, mode)
                generated_y.append(generation_per)
            generated_y=np.asarray(generated_y)
            
            real_y=real_y*self.dataset.std+self.dataset.mean
            generated_y=generated_y*self.dataset.std+self.dataset.mean
            
            self.save_generation(real_y, generated_y, save_path)
            print('CRPS generatioin on {} dataset finished'.format(mode))
        else:
            print('generation already exists, loading generation from {}'.format(save_path))
            real_y_file=os.path.join(save_path, 'real_y.npy')
            generated_y_file=os.path.join(save_path, 'generated_y.npy')
            real_y=np.load(real_y_file)
            generated_y=np.load(generated_y_file)
            print('generation loading on {} dataset finished'.format(mode))
        
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
    def save_score(self, score, mode='validation', metric='CRPS'):
        assert mode in ['validation', 'test']
        if mode =='validation':
            path=self.args.validation_path
        else:
            path=self.args.test_path
            
        score_file=os.path.join(self.args.dataset, self.model_dir, path, '{}_{}'.format(self.args.num_sampling,self.args.num_experiments), '{}.txt'.format(metric))
        with open (score_file,'w') as f:
            f.write(str(score))
        print('score {} on the {} dataset saved to {}'.format(score, mode, score_file))
        
def train(datase, args):
    
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
    
        model= DMDNN_model(args, device)   
        load_path=os.path.join(model.args.dataset, model.model_dir, model.args.save_model_path)
        if not os.path.exists(load_path):
            print('load path {} does not exist, training from scratch'.format(load_path))
            train_losses=model.train()
            model.save_model()
        else:
            print('load path exists, loading...')
            model.load_model()
    
        #### validation on the validation set##########
        real_y, final_generation=model.CRPS_generation(args.num_sampling, args.num_experiments, mode='validation')  ## shape of num_experiments, num_sampling, N, dim
        CRPS_score=CRPS(real_y, final_generation, average=args.average)
        model.save_score(CRPS_score, mode='validation', metric='CRPS')
        real_y, final_generation=model.CRPS_generation(args.num_sampling, args.num_experiments, mode='validation')  ## shape of num_experiments, num_sampling, N, dim
        RMSE_score=RMSE(real_y, final_generation, average=args.average)
        model.save_score(RMSE_score, mode='validation', metric='RMSE')
        
        #    #test on the test dataset
        real_y, final_generation=model.CRPS_generation(args.num_sampling, args.num_experiments, mode='test')  ## shape of num_experiments, num_sampling, N, dim
        CRPS_score=CRPS(real_y, final_generation, average=args.average)
        model.save_score(CRPS_score, mode='test',metric='CRPS')  
        real_y, final_generation=model.CRPS_generation(args.num_sampling, args.num_experiments, mode='test')  ## shape of num_experiments, num_sampling, N, dim
        RMSE_score=RMSE(real_y, final_generation, average=args.average)
        model.save_score(RMSE_score, mode='test', metric='RMSE')  
        

if __name__=='__main__':
    
    import argparse
    parser=argparse.ArgumentParser(description='DMDNN for multivariate time series generation')
   
    parser.add_argument('--ratio_list',nargs='+',type=int, default=[8,1,1])
    parser.add_argument('--train_length',type=int, default=50)
    parser.add_argument('--timesteps',type=int, default=40)
    parser.add_argument('--pred_length', type=int, default=1, help='10 minutes for one step')
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--norm',type=bool,default=True)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dataset', type=str, default='GOOG', help= 'AE or PC or PM or WP')
    parser.add_argument('--data_path', type=str, default='../dataset/H1-01F.xlsx', help='relative path to save the csv data')
    parser.add_argument('--data_length',type=int,default=100000) # PC 2075259, #PM  43824  #AE 19735 #WP 20432 
    
    parser.add_argument('--num_Gaussians', type=int, default=5)
    parser.add_argument('--inputs_index',nargs='+', type=str, default=['Appliances','Windspeed'])
    parser.add_argument('--num_inputs', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=1)


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
    parser.add_argument('--num_experiments', type=int, default=2,
                        help='number of experiments to be conducted')
    parser.add_argument('--num_sampling', type=int, default=500,
                        help='number of samplings for each test samples')
    
    parser.add_argument("--rnn_hidden", type=int, default=32, 
                        help="number of hidden units in the GRU")     
    parser.add_argument("--rnn_layers", type=int, default=4, 
                        help="number of hidden layers in the GRU")  
    parser.add_argument("--dropout", type=float, default=0.0, 
                        help="output dropout in the GRU") 
    
    
    parser.add_argument('--num_channels', nargs='+', type=int, default=[32]*2)
    parser.add_argument('--h_dim', type=int, default=32)
    
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help='kernel size to use') 
    parser.add_argument('--dilation_size', type=int, default=1, 
                        help='dilation size to use')  

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
        
#    dataset_list=['BeijingPM']
    for dataset in dataset_list:
        args.dataset=dataset
        train(dataset, args)    
    
    
    
    
    
    
    
    
#    batch=64
#    num_inputs=1
#    length=41
#    kernel_size=3
#    num_Gaussians=5
#    num_channels=[32,32]
#    rnn_hidden=32
#    rnn_layers=4
#    num_outputs=1
#    h_dim=32
#    dilation_size=1
#    dropout=0.1
#    
#    
#    x=torch.normal(0,1,size=(batch, num_inputs, length))
#    
#    layer=Conv1dNet(num_inputs, num_channels)
#    
#    
#    a=torch.sign(compute_gradient_penalty(layer, x)).detach()
    
   
    
#    DMDNN_layer=DMDNN(num_Gaussians, num_inputs, num_channels, rnn_hidden, rnn_layers, num_outputs, h_dim, kernel_size, dilation_size, dropout)
#
#    pi, mu, sigma=DMDNN_layer(x)
    
