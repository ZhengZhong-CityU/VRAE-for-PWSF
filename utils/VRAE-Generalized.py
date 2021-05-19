# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:56:35 2020

@author: zhongzheng
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from VRAETP_load_data import Read_data, random_sample
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import itertools
from evaluations import CRPS, RMSE
import torch.nn.functional as F
from torch_reparameterization import torch_gamma_rp, torch_beta_rp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def KL_Gamma(alpha1, beta1, alpha2, beta2):
    kl=(alpha1-1)*torch.digamma(alpha1)+torch.log(beta1)-alpha1-torch.lgamma(alpha1)
    +torch.lgamma(alpha2)-alpha2*torch.log(beta2)-(alpha2-1)(torch.digamma(alpha1)-torch.log(beta1))+alpha1*beta2/beta1
    return kl

def KL_Beta(alpha1, beta1, alpha2, beta2):
    kl=torch.log(alpha1+beta1)+torch.log(alpha2)+torch.log(beta2)-torch.log(alpha2+beta2)-torch.log(alpha1)-torch.log(beta1)
    +(alpha1-alpha2)*(torch.digamma(alpha1)-torch.digmma(alpha1+beta1))+(beta1-beta2)*(torch.digamma(beta1)-torch.digmma(alpha1+beta1))
    return kl
    
    
class MVRAEGMM(nn.Module):
    def __init__(self, num_Gaussians, x_dim, h_dim, z_dim, rnn_layers, posterior='Gaussian'):
    ### x_dim is the dim of the inputs used for prediction
    ### h_dim is the dim of the hidden state
    ### z_dim is the sampling satate
    ### we assum a GMM for the observation model
        super(MVRAEGMM, self).__init__()
        
        self.num_Gaussians=num_Gaussians
        self.x_dim=x_dim
        self.h_dim=h_dim
        self.z_dim=z_dim
        self.rnn_layers=rnn_layers
        self.posterior=posterior

        
        ##feature extrating trasformations
        self.phi_x = nn.Sequential(nn.Linear(self.x_dim, self.h_dim),nn.ReLU(),nn.Linear(self.h_dim, self.h_dim),nn.ReLU())
        self.phi_z=nn.Sequential(nn.Linear(self.z_dim, self.h_dim),nn.ReLU(),nn.Linear(self.h_dim, self.h_dim),nn.ReLU())
        
        ## Prior network for hidden dim of x to z dim of z
        self.prior = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),nn.ReLU(),nn.Linear(self.h_dim, self.h_dim),nn.ReLU())
        
        ## for Gaussian, para1 and para2 indicate mean and std, for Beta or Alpha, para1 and para2 indicate alpha and beta
        if self.posterior=='Gaussian':
            self.prior_para1=nn.Linear(self.h_dim, self.z_dim)

        else:
            self.prior_para1=nn.Sequential(nn.Linear(self.h_dim, self.z_dim),nn.Softplus())
        self.prior_para2=nn.Sequential(nn.Linear(self.h_dim, self.z_dim),nn.Softplus())
            
        
        ## Decoder, generation from the inputs hidden dim and z to x_dim, observation model is a GMM
        self.decoder=nn.Sequential(nn.Linear(self.h_dim + self.h_dim, self.h_dim),nn.ReLU(),nn.Linear(self.h_dim, self.h_dim),nn.ReLU())
        self.decoder_pi=nn.Sequential(nn.Linear(self.h_dim, self.num_Gaussians),nn.Softmax(dim=-1))
        self.decoder_mean=nn.Linear(self.h_dim,int(self.num_Gaussians*self.x_dim))
        self.decoder_std=nn.Sequential(nn.Linear(self.h_dim, int(self.num_Gaussians*self.x_dim)),nn.Softplus())

        ## Encoder, inference from the inputs hidden dim and y to obtain the posterior of z
        self.encoder=nn.Sequential(nn.Linear(self.h_dim + self.h_dim, self.h_dim),nn.ReLU(),nn.Linear(h_dim, h_dim),nn.ReLU())
        
        ## for Gaussian, para1 and para2 indicate mean and std, for Beta or Alpha, para1 and para2 indicate alpha and beta
        if self.posterior=='Gaussian':
            self.encoder_para1=nn.Sequential(nn.Linear(self.h_dim, self.h_dim),nn.ReLU(),nn.Linear(self.h_dim, self.z_dim))
        else:
            self.encoder_para1=nn.Sequential(nn.Linear(self.h_dim, self.h_dim),nn.ReLU(),nn.Linear(self.h_dim, self.z_dim),nn.Softplus())
        self.encoder_para2=nn.Sequential(nn.Linear(self.h_dim, self.z_dim),nn.Softplus())
        
        
        ## Recurrence from h_(t-1), z_t and x_t to h_t
        self.rnn=nn.GRU(self.h_dim+self.h_dim, self.h_dim, self.rnn_layers, batch_first=True)
        
    def forward(self, x):
        ## x represents the history, shape of batch, x_dim, length
        
        ## initial hidden state
        h = Variable(torch.zeros(self.rnn_layers, x.shape[0], self.h_dim)).to(device) 
        
        kld_loss=0
        NLL_GMM=0
        
        for t in range(x.shape[-1]):
            ## Extract the x features 
            phi_x_t=self.phi_x(x[:,:,t])
            
            ## Prior
            prior_t=self.prior(h[-1])
            prior_para1_t=self.prior_para1(prior_t)
            prior_para2_t=self.prior_para2(prior_t)
            
            ## Encoder
            encoder_t=self.encoder(torch.cat([phi_x_t, h[-1]],-1))
            encoder_para1_t=self.encoder_para1(encoder_t)
            encoder_para2_t=self.encoder_para2(encoder_t)
            
            ## Decoder
            # Sampling from the posterior
            z_t=self.reparameterized_sample(encoder_para1_t, encoder_para2_t)
            phi_z_t=self.phi_z(z_t)
            
            decoder_t=self.decoder(torch.cat([phi_z_t,h[-1]],-1))
            decoder_pi_t=self.decoder_pi(decoder_t)
            decoder_mean_t=self.decoder_mean(decoder_t).reshape(-1,self.num_Gaussians,self.x_dim) 
            decoder_std_t=self.decoder_std(decoder_t).reshape(-1,self.num_Gaussians,self.x_dim)
            
            ## Recurrence
            # RNN inputs should be shape of [batch, length, dim], return raw_outputs [batch, length, hidden], state [rnn_layer, batch, dim]
            if t<x.shape[-1]-1:
                _, h=self.rnn(torch.cat([phi_x_t,phi_z_t],-1).unsqueeze(1),h)
        
            ## Computing loss
            kld_loss+=self.kld_loss(encoder_para1_t,encoder_para2_t,prior_para1_t,prior_para2_t)
            NLL_GMM+=self.NLL_GMM(x[:,:,t],decoder_pi_t,decoder_mean_t,decoder_std_t)
        
        return kld_loss, NLL_GMM
          
    # sample by reparameterization
    def reparameterized_sample(self, para1, para2):
        if self.posterior=='Gaussian':
            z=Variable(torch.randn(para1.shape)).to(device)
            sample_z=z*para2+para1
        if self.posterior=='Gamma':
            sample_z=torch_gamma_rp(para1, para2).to(device)
        if self.posterior=='Beta':
            sample_z=torch_beta_rp(para1, para2).to(device)            
  
        return sample_z
    
    def prediction(self, x, pred_length):
        ### x isthe input used for prediction, shape of batch, dim, length 
        
        ## Summarize the input sequence x, sample z from the posterior at each time point
        # Initial hidden state
        h = Variable(torch.zeros(self.rnn_layers, x.shape[0], self.h_dim)).to(device)         
        for t in range(x.shape[-1]):
            # Extract the x features 
            phi_x_t=self.phi_x(x[:,:,t])
            
            # Encoder
            encoder_t=self.encoder(torch.cat([phi_x_t, h[-1]],-1))
            encoder_para1_t=self.encoder_para1(encoder_t)
            encoder_para2_t=self.encoder_para2(encoder_t)

            # Sampling from the posterior
            z_t=self.reparameterized_sample(encoder_para1_t, encoder_para2_t)
            phi_z_t=self.phi_z(z_t)
            
            # Recurrence
            _, h=self.rnn(torch.cat([phi_x_t,phi_z_t],-1).unsqueeze(1),h)
        
        ## Prediction initilized by the hidden state summarized from x, sampel z from the prior at each time point
        samples=[]
        for t in range(pred_length):
            # Prior
            prior_t = self.prior(h[-1])
            prior_para1_t = self.prior_para1(prior_t)
            prior_para2_t = self.prior_para2(prior_t)
            

			#sampling from the prior
            z_t = self.reparameterized_sample(prior_para1_t, prior_para2_t)
            phi_z_t = self.phi_z(z_t)
			
			# Decoder
            decoder_t = self.decoder(torch.cat([phi_z_t, h[-1]], -1))
            decoder_pi_t=self.decoder_pi(decoder_t)
            decoder_mean_t = self.decoder_mean(decoder_t).reshape(-1,self.num_Gaussians,self.x_dim) 
            decoder_std_t = self.decoder_std(decoder_t).reshape(-1,self.num_Gaussians,self.x_dim) 
            
            # Sampling from the GMM model to obtain the prediction at each time point
            x_t=self.GMM_sample(decoder_pi_t, decoder_mean_t, decoder_std_t)
            
            # Feature extraction of the inputs
            phi_x_t = self.phi_x(x_t)  # shape [batch, x_dim]

			# Recurrence only if we still have some points to predict
            if t<pred_length-1:
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], -1).unsqueeze(1), h)

            samples.append(x_t)
            
        return torch.stack(samples, -1)  ## shape [batch, dim, pred_length]
         
    def GMM_sample(self, pi, mu, sigma):
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
        
    def kld_loss(self, encoder_para1, encoder_para2, prior_para1, prior_para2):
        if self.posterior=='Gaussian':
            loss=0.5*torch.sum((2 * torch.log(prior_para2) - 2 * torch.log(encoder_para2) + 
			(encoder_para2.pow(2) + (encoder_para1 - prior_para1).pow(2)) /
			prior_para2.pow(2) - 1)) 
        else:

            if self.posterior=='Beta':
                loss=torch.sum(KL_Beta(encoder_para1, encoder_para2, prior_para1, prior_para2))
            if self.posterior=='Gamma':
                loss=torch.sum(KL_Gamma(encoder_para1, encoder_para2, prior_para1, prior_para2))
        return	loss
    
    def NLL_GMM(self, y, pi, mu, sigma):
        # y: shape of batch, y_dim
        # pi: batch, num_Gaussians
        # mu: batch, num_Gaussians, y_dim
        # sigma: batch, num_Gaussians,y_dim
        N, output_dim=y.shape
        t_y=y.unsqueeze(1)
        t_pi=pi.unsqueeze(-1)
        mixture=torch.distributions.normal.Normal(mu, sigma)
        log_prob=mixture.log_prob(t_y)
        #    print(log_prob.shape)
        weighted_logprob=log_prob+torch.log(t_pi)
        log_sum=torch.logsumexp(weighted_logprob, dim=1)
        return -torch.sum(log_sum)
    
class MMVRAEGMM_model():
    def __init__(self, args, device):
        self.args=args
        self.device=device
        self.dataset=Read_data(self.args)
        self.build()
        self.trainable_num=self.get_parameter_number()               
        
    def build(self):
        self.network=MVRAEGMM(self.args.num_Gaussians,self.args.num_inputs, self.args.h_dim, self.args.z_dim, self.args.rnn_layers)
        self.network=self.network.to(self.device)    
        self.optimizer=torch.optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
    
    @property
    def model_dir(self):
        return '{}_{}_{}_{}_{}_{}_{}'.format(self.args.posterior,self.args.train_length, self.args.num_Gaussians, self.args.z_dim, self.args.h_dim, self.args.lr, self.args.iterations) 

    def get_parameter_number(self):
        trainable_num = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print('Number of trainable parameters: {}; '.format(trainable_num))
        return trainable_num         
        
    def train(self):
        print('network training start!')
        self.network.train()
        train_losses=[]
        
        for iteration in range(self.args.iterations):
            real_samples=random_sample(self.dataset.train_np,self.args.batch_size).type(torch.FloatTensor)  ## shape of batch, dim, train_length
            real_samples=real_samples.to(self.device)
            self.optimizer.zero_grad()
            kld, nll=self.network(real_samples)
            loss=kld+nll
            loss.backward()
            self.optimizer.step()
            loss_value=loss.data.item()
            train_losses.append(loss_value)
            
            print('training iteration {}|{}: NLL loss: {}, kld loss: {}'.format(iteration+1, self.args.iterations, nll.data.item(), kld.data.item()))
        self.save_losses(train_losses)
        return train_losses 

    def prediction(self, x, pred_length):
        self.network.eval()
        
        x=x.to(self.device)        
        fake_y=self.network.prediction(x, pred_length)
        predicted_y=fake_y.cpu().detach().numpy()
        return  predicted_y 

    def prediction_eval(self, num_sampling, mode='validation'):
        ### since we will test on the testset for many times, the num_sampling indicates the total number of trials per experiments
        
        assert mode in ['validation', 'test']
        if mode=='validation':
            dataset= self.dataset.val_loader
        else:
            dataset=self.dataset.test_loader
        
        print('VRAETP prediction on the {} dataset start!'.format(mode))  
        prediction_per=[]  ## generation of each experiment
        real_y=[]
        for batch_id, samples in enumerate(dataset):
            Time_prediction=[]
            test_samples=samples[0].type(torch.FloatTensor)
            x=test_samples[:,:,:self.args.timesteps]
            y=test_samples[:,:,self.args.timesteps:].detach().numpy()
            real_y.append(y)
            for time in range(num_sampling):
                print('times {}|{} batches {}|{}'.format(time+1, num_sampling, batch_id+1, len(self.dataset.test_loader)))
                predicted_y=self.prediction(x, self.args.pred_length)
                Time_prediction.append(predicted_y)              
            Time_prediction=np.asarray(Time_prediction)
            prediction_per.append(Time_prediction)
        predicted_y=np.concatenate(prediction_per,1)
        real_y=np.concatenate(real_y,0)        
        return real_y, predicted_y    
    
    def CRPS_prediction(self, num_sampling=100, num_experiments=2, mode='validation'):
        '''
        for CRPS evaluation, we need to conduct a number of experiments, for each experiments, we need to sample num_sampling points
        '''
        
        assert mode in ['validation', 'test']
        if mode =='validation':
            path=self.args.validation_path
        else:
            path=self.args.test_path
        
        save_path=os.path.join(self.args.dataset, self.model_dir, path, '{}_{}_{}_{}'.format(num_sampling, num_experiments, self.args.timesteps, self.args.pred_length))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            predicted_y=[]
            for index in range(num_experiments):
                real_y, prediction_per=self.prediction_eval(num_sampling, mode)
                predicted_y.append(prediction_per)
            predicted_y=np.asarray(predicted_y)
            
            real_y=real_y*self.dataset.std+self.dataset.mean
            predicted_y=predicted_y*self.dataset.std+self.dataset.mean
            
            self.save_prediction(real_y, predicted_y, save_path)
            print('CRPS prediction on {} dataset finished'.format(mode))
        else:
            print('prediction already exists, loading prediction from {}'.format(save_path))
            real_y_file=os.path.join(save_path, 'real_y.npy')
            predicted_y_file=os.path.join(save_path, 'generated_y.npy')
            real_y=np.load(real_y_file)
            predicted_y=np.load(predicted_y_file)
            print('prediction loading on {} dataset finished'.format(mode))
        
        return real_y, predicted_y
            
     
    def save_prediction(self, real_y, predicted_y, save_path):
        print('real samples shape {}; predicted samples shape: {}'.format(real_y.shape, predicted_y.shape))
        real_y_file=os.path.join(save_path, 'real_y.npy')
        np.save(real_y_file, real_y) 
        predicted_y_file=os.path.join(save_path, 'generated_y.npy')
        np.save(predicted_y_file, predicted_y)   
        print('successfully save the prediction to {}'.format(save_path))
            
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
    def save_score(self, score, mode='validation',  metric='CRPS'):
        assert mode in ['validation', 'test']
        if mode =='validation':
            path=self.args.validation_path
        else:
            path=self.args.test_path
            
        score_file=os.path.join(self.args.dataset, self.model_dir, path, '{}_{}_{}_{}'.format(self.args.num_sampling,self.args.num_experiments, self.args.timesteps, self.args.pred_length), '{}.txt'.format(metric))
        with open (score_file,'w') as f:
            f.write(str(score))
        print('score {} on the {} dataset saved to {}'.format(score, mode, score_file))
        
 

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='VAE for time series prediction')
    parser.add_argument('--ratio_list',nargs='+',type=int, default=[8,1,1])
    parser.add_argument('--train_length',type=int, default=41)
    parser.add_argument('--timesteps',type=int, default=40)
    parser.add_argument('--pred_length', type=int, default=1, help='10 minutes for one step')
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--norm',type=bool,default=True)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dataset', type=str, default='BeijingPM', help= 'AE or PC or PM or WP')
    parser.add_argument('--data_path', type=str, default='../dataset/H1-01F.xlsx', help='relative path to save the csv data')
    parser.add_argument('--data_length',type=int,default=100000) # PC 2075259, #PM  43824  #AE 19735 #WP 20432 
    
    parser.add_argument('--num_Gaussians', type=int, default=5)
    parser.add_argument('--inputs_index',nargs='+', type=str, default=['Appliances'])
    parser.add_argument('--num_inputs', type=int, default=1)

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
    parser.add_argument("--rnn_layers", type=int, default=4, 
                        help="number of hidden layers in the GRU")  

    parser.add_argument('--average', type=bool, default=True,
                        help='whether to average the CRPS or MAE score') 
    
    parser.add_argument('--h_dim', type=int, default=32, help='hidden dimension to be used')
    parser.add_argument('--z_dim', type=int, default=16, help='noise dimension')
    
    parser.add_argument('--posterior', type=str, default='Gamma', help='Gaussian or Gamma or Beta')
    
    
    
    
    args=parser.parse_args()
    
    assert args.posterior in ['Gaussian', 'Gamma', 'Beta']
    
    if args.dataset=='AE':
        args.data_path='../dataset/energydata_complete.csv' 
        args.inputs_index=['Windspeed']
        
    if 'PM' in args.dataset:
        args.data_path='../dataset/{}20100101_20151231.csv'.format(args.dataset)
        args.inputs_index=['Iws']
           
    args.num_inputs=len(args.inputs_index)
    args.train_length=args.timesteps+args.pred_length

    #specify the seed and device
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed) 
        
    z_dim_list=[4, 8]
    
    for z_dim in z_dim_list:
        args.z_dim=z_dim
    
        model= MMVRAEGMM_model(args, device)   
        load_path=os.path.join(model.args.dataset, model.model_dir, model.args.save_model_path)
        if not os.path.exists(load_path):
            print('load path {} does not exist, training from scratch'.format(load_path))
            train_losses=model.train()
            model.save_model()
        else:
            print('load path exists, loading...')
            model.load_model()
    
        #### validation on the validation set##########
        real_y, final_prediction=model.CRPS_prediction(args.num_sampling, args.num_experiments, mode='validation')  ## shape of num_experiments, num_sampling, N, dim
        CRPS_score=CRPS(real_y, final_prediction, average=args.average)
        model.save_score(CRPS_score, mode='validation', metric='CRPS')
        real_y, final_prediction=model.CRPS_prediction(args.num_sampling, args.num_experiments, mode='validation')  ## shape of num_experiments, num_sampling, N, dim
        RMSE_score=RMSE(real_y, final_prediction, average=args.average)
        model.save_score(RMSE_score, mode='validation', metric='RMSE')
        #    #test on the test dataset
        real_y, final_prediction=model.CRPS_prediction(args.num_sampling, args.num_experiments, mode='test')  ## shape of num_experiments, num_sampling, N, dim
        CRPS_score=CRPS(real_y, final_prediction, average=args.average)
        model.save_score(CRPS_score, mode='test',metric='CRPS')   
        real_y, final_prediction=model.CRPS_prediction(args.num_sampling, args.num_experiments, mode='test')  ## shape of num_experiments, num_sampling, N, dim
        RMSE_score=RMSE(real_y, final_prediction, average=args.average)
        model.save_score(RMSE_score, mode='test', metric='RMSE')  
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        