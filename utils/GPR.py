# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 09:48:32 2020

@author: zhongzheng
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn

from VRAETP_load_data import Read_data, random_sample
import matplotlib.pyplot as plt
from evaluations import CRPS, RMSE
import itertools
import gpytorch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)                  

          
        
class GP():
    def __init__(self, args, device):
        self.args=args
        self.device=device
        self.dataset=Read_data(self.args)


    @property
    def model_dir(self):
        return 'GP_{}_{}_{}_{}'.format(self.args.timesteps, self.args.pred_length,self.args.lr, self.args.iterations) 
    
        
    def train_GP(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)
        
        model.train()
        likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        
        for i in range(self.args.iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
#            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f  outputscale: %.3f noise: %.3f' % (
#                    i + 1, self.args.iterations, loss.item(),
#                    model.covar_module.base_kernel.lengthscale.item(),
#                    model.covar_module.outputscale.item(),
#                    model.likelihood.noise.item()
#                    ))
            optimizer.step()
                   
        return model, likelihood

    def prediction(self, model, likelihood, num_sampling):
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor([self.args.timesteps+1],dtype=torch.float)
            observed_pred = likelihood(model(test_x))
            generated_y=observed_pred.sample(sample_shape=torch.Size((num_sampling,)))

        return generated_y.numpy()  ## [num_sampling, 1]


    def pred_eval(self, num_sampling=100, num_experiments=2, mode='validation'):
        ### since we will test on the testset for many times, the num_sampling indicates the total number of trials per experiments
        
        assert mode in ['validation', 'test']
        if mode=='validation':
            dataset= self.dataset.val_np
            path=self.args.validation_path
        else:
            dataset=self.dataset.test_np
            path=self.args.test_path
        
        save_path=os.path.join(self.args.dataset, self.model_dir, path, '{}_{}'.format(self.args.num_sampling,self.args.num_experiments))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            generated_y=[]
            real_y=[]
            
            for i in range(dataset.shape[0]):
                s_time=time.time()
                x=dataset[i,0,0:self.args.timesteps]  ## [timesteps,]
                y=dataset[i,0,-1] ## []
                real_y.append(y)
                
                generated_y_t=[]
                for t in range(num_experiments):
                    train_x=torch.tensor(np.arange(1,self.args.timesteps+1),dtype=torch.float)
                    train_y=torch.tensor(x,dtype=torch.float)
                    model, likelihood=self.train_GP(train_x, train_y)

                    pred_y=self.prediction(model, likelihood, num_sampling)
                    
                    generated_y_t.append(pred_y)
                    
                e_time=time.time()
                generated_y_t=np.asarray(generated_y_t)  ## [num_experiment, num_sampling, 1]
                
                print('sample {}|{}: time: {} '.format(i+1, dataset.shape[0], e_time-s_time))
                
                generated_y.append(generated_y_t)
            
            
            generated_y=np.asarray(generated_y).transpose(1,2,0,3)    ## [num_experiments, num_sampling, N, 1]
            real_y=np.asarray(real_y)[:,None]
            
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
        model=GP(args, device)    

        ### test stage###
        real_y, final_generation=model.pred_eval(args.num_sampling, args.num_experiments,'test')
        
    
        MAE_score=CRPS(real_y, final_generation, average=args.average)
        model.save_score(MAE_score, mode='test', metric='CRPS')
        print(MAE_score)
        

        RMSE_score=RMSE(real_y, final_generation, average=args.average)
        model.save_score(RMSE_score, mode='test', metric='RMSE')
        print(RMSE_score)
        

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
    parser.add_argument('--dataset', type=str, default='GOOG', help= 'AE or PC or PM or WP')
    parser.add_argument('--data_path', type=str, default='../dataset/H1-01F.xlsx', help='relative path to save the csv data')
    parser.add_argument('--data_length',type=int,default=100000) # PC 2075259, #PM  43824  #AE 19735 #WP 20432 
    

    parser.add_argument('--inputs_index',nargs='+', type=str, default=['Appliances','Windspeed'])
    parser.add_argument('--num_inputs', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.1, 
                        help="adam: learning rate, default=0.1")
    parser.add_argument("--iterations", type=int, default=50, 
                        help="number of iterations of training, default=50") 
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
#    dataset_list=['GuangzhouPM', 'ShenyangPM']
    
    for dataset in dataset_list:
        args.dataset=dataset
        train(dataset, args)
        
 
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        