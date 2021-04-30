# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:16:57 2020

@author: zhongzheng
"""


import torch
import numpy as np
from VRAETP_load_data import Read_data, random_sample
import os
from evaluations import MAE, RMSE
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
class MPersistent():
    def __init__(self, args, device):
        self.args=args
        self.device=device
        self.dataset=Read_data(self.args)
        assert self.args.method in ['mean', 'nearest', 'median']
        
    @property    
    def model_dir(self):
        return '{}_{}_{}'.format(self.args.method,self.args.timesteps, self.args.pred_length) 
 
    def prediction(self,x):
        ## x of shape batch, dim, timesteps
        x=x.to(self.device)
        if self.args.method=='mean':
            pred_y=torch.mean(x,dim=-1)
        if self.args.method=='nearest':
            pred_y=x[:,:,-1]
        if self.args.method=='median':
            pred_y=torch.median(x,dim=-1)[0]
        
        pred_y=pred_y.unsqueeze(-1)
        pred_y=pred_y.repeat(1,1,self.args.pred_length)
        
#        print(pred_y.shape)
            
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
        
        print('{} generation on the {} dataset start!'.format(self.args.method, mode)) 

        save_path=os.path.join(self.args.dataset, self.model_dir, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            generated_y=[]
            real_y=[]
            for batch_id, samples in enumerate(dataset):
                test_samples=samples[0].type(torch.FloatTensor)
                x=test_samples[:,:,:self.args.timesteps]
                y=test_samples[:,:,self.args.timesteps:].detach().numpy()
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
        
    pred_length_list=[1] ## ranging from ten minutes to 60 minutes
    scores=[]
    for pred_length in pred_length_list:
        args.pred_length=pred_length
        args.train_length=args.timesteps+args.pred_length    
        model=MPersistent(args, device)    

        ### test stage###
        real_y, final_generation=model.pred_eval('test')
        
#        ## if not denormalized, do this part
#        real_y=real_y*model.dataset.std+model.dataset.mean
#        final_generation=final_generation*model.dataset.std+model.dataset.mean
#        path=model.args.test_path
#        save_path=os.path.join(model.args.dataset, model.model_dir, path)
#        model.save_generation(real_y, final_generation, save_path)
        
        MAE_score=MAE(real_y, final_generation, average=args.average)
        scores.append(MAE_score)
        model.save_score(MAE_score, mode='test', metric='MAE')
        
        RMSE_score=RMSE(real_y, final_generation, average=args.average)
        scores.append(RMSE_score)
        model.save_score(RMSE_score, mode='test', metric='RMSE')
    print(scores)
        
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Predictive ANN for multivariate time series generation')
    parser.add_argument('--method', choices=['mean','median','nearest'], default='nearest')
    parser.add_argument('--ratio_list',nargs='+',type=int, default=[8,1,1])
    parser.add_argument('--train_length',type=int, default=50)
    parser.add_argument('--timesteps',type=int, default=40)
    parser.add_argument('--pred_length', type=int, default=1, help='10 minutes for one step')
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--norm',type=bool,default=True)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dataset', type=str, default='', help= 'AE or PC or PM or WP')
    parser.add_argument('--data_path', type=str, default='../dataset/H1-01F.xlsx', help='relative path to save the csv data')
    parser.add_argument('--data_length',type=int,default=100000) # PC 2075259, #PM  43824  #AE 19735 #WP 20432 
    
    parser.add_argument('--inputs_index',nargs='+', type=str, default=['Appliances','Windspeed'])

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')   
    parser.add_argument('--validation_path', type=str, default='validation_generation', 
                        help='path to save the generation on the validation set')
    parser.add_argument('--test_path', type=str, default='test_generation', 
                        help='path to save the generation on the test set') 
    
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
    

    
