# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:56:53 2020

@author: zhongzheng
"""

import pandas as pd
import torch
import numpy as np
from VRAETP_load_data import Read_data, random_sample
import os
from evaluations import MAE, RMSE
import random
from pmdarima.arima import auto_arima
import time




class ARIMA_model():
    def __init__(self, args, device):
        self.args=args
        self.device=device
        self.dataset=Read_data(self.args)
        
    @property    
    def model_dir(self):
        return 'ARIMA_{}_{}'.format(self.args.timesteps, self.args.pred_length) 
    
    def pred_eval(self, mode='validation'):
        
        assert mode in ['validation', 'test']
        
        if mode=='validation':
            dataset= self.dataset.val_np
            path=self.args.validation_path
        else:
            dataset=self.dataset.test_np
            path=self.args.test_path
        
        print('ARIMA generation on the {} dataset start!'.format(mode)) 

        save_path=os.path.join(self.args.dataset, self.model_dir, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            generated_y=[]
            real_y=[]            
            for i in range(dataset.shape[0]):
                s_time=time.time()
                x=dataset[i,0,0:self.args.timesteps]
                stepwise_model = auto_arima(x, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=1,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
                                   
                y=dataset[i,0,-1]
                real_y.append(y)
                pred_y=stepwise_model.predict(n_periods=1)
                generated_y.append(pred_y)
                e_time=time.time()
                print('sample {}|{}: real_y: {} pred_y:{} time: {}'.format(i+1, dataset.shape[0], y, pred_y, e_time-s_time))
            
            
            generated_y=np.concatenate(generated_y,0)  
            real_y=np.asarray(real_y)
            
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
    for pred_length in pred_length_list:
        args.pred_length=pred_length
        args.train_length=args.timesteps+args.pred_length    
        model=ARIMA_model(args, device)  

        ### test stage###
        real_y, generated_y=model.pred_eval('test')
        
#        ## if not denormalized, do this part
#        real_y=real_y*model.dataset.std+model.dataset.mean
#        generated_y=generated_y*model.dataset.std+model.dataset.mean
#        path=model.args.test_path
#        save_path=os.path.join(model.args.dataset, model.model_dir, path)
#        model.save_generation(real_y, generated_y, save_path)
        
        MAE_score=MAE(real_y, generated_y, average=args.average)
        model.save_score(MAE_score, mode='test', metric='MAE')
        

        RMSE_score=RMSE(real_y, generated_y, average=args.average)
        model.save_score(RMSE_score, mode='test', metric='RMSE')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Predictive ANN for multivariate time series generation')
    parser.add_argument('--ratio_list',nargs='+',type=int, default=[8,1,1])
    parser.add_argument('--train_length',type=int, default=50)
    parser.add_argument('--timesteps',type=int, default=40)
    parser.add_argument('--pred_length', type=int, default=1, help='10 minutes for one step')
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--norm',type=bool,default=True)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dataset', type=str, default='AE', help= 'AE or PC or PM or WP')
    parser.add_argument('--data_path', type=str, default='../dataset/energydata_complete.csv', help='relative path to save the csv data')
    parser.add_argument('--data_length',type=int,default=100000) # PC 2075259, #PM  43824  #AE 19735 #WP 20432 
    
    parser.add_argument('--inputs_index',nargs='+', type=str, default=['Windspeed'])

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
#    model=ARIMA_model(args, device)
#    real_y, generated_y=model.pred_eval('validation')
#    MAE_score=MAE(real_y, generated_y, average=args.average)
#    print(MAE_score)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    