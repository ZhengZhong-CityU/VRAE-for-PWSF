# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:52:15 2020

@author: zhongzheng
"""
import numpy as np

def CRPS(real_y, generated_y, average=True):
    '''
    args:
        real_y, should be shape of [N, dim], represents the target to be predicted
        generated_y, should be shape of [num_experiments, num_sampling, N, dim], represents the generation results.
        num_experiments is the number of experiments, default 2
        num_sampling is the number of sampling in each experiments, default 100
    '''
#    real_y=real.expand_dims((0,1))
    print(real_y.shape)
    print(generated_y.shape)
    if average:
        CRPS_score=np.mean(np.abs(generated_y[0]-real_y)-0.5*np.abs(generated_y[0]-generated_y[1]))
    else:
        CRPS_score=np.mean(np.abs(generated_y[0]-real_y)-0.5*np.abs(generated_y[0]-generated_y[1]),axis=(0,1))
    return CRPS_score


def MAE(real_y, generated_y, average=True):
    '''
    args: both real_y and generated_y should be the shape of N, dim
        
    '''
    if average:
        
        MAE_score=np.mean(np.abs(real_y-generated_y))
    else:
        MAE_score=np.mean(np.abs(real_y-generated_y), axis=0)
    return MAE_score


def RMSE(real_y, generated_y, average=True):
    '''
    args:
        real_y, should be shape of [N, dim], represents the target to be predicted
        generated_y, should be shape of [num_experiments, num_sampling, N, dim], represents the generation results.
        num_experiments is the number of experiments, default 2
        num_sampling is the number of sampling in each experiments, default 100
    
    '''
    
    if generated_y.ndim>2:
        if average:
#            RMSE_score=np.mean(np.sqrt(np.mean(pow(real_y-generated_y,2), axis=(2,3))))
#            RMSE_score=np.sqrt(np.mean(pow(real_y-generated_y[0,0,:,:],2)))
            RMSE_score=np.sqrt(np.mean(pow(np.mean(generated_y, axis=(0,1))-real_y,2)))
        else:
            RMSE_score=np.sqrt(np.mean(pow(np.mean(generated_y, axis=(0,1))-real_y,2), axis=0))
        
    else:   
        if average:
            RMSE_score=np.sqrt(np.mean(pow(real_y-generated_y,2)))
        else:
            RMSE_score=np.sqrt(np.mean(pow(real_y-generated_y,2), axis=0))  
        
        
    return RMSE_score
    
def bound(generated_y, confidence=0.95):
    '''
    generated_y, should be shape of [num_experiments, num_sampling, N, dim], represents the generation results.
    '''
#    print(generated_y.shape)
    alpha=(1-confidence)/2
    generated_y=generated_y.reshape(-1,generated_y.shape[2])
    ub=np.quantile(generated_y,1-alpha, axis=0)
    lb=np.quantile(generated_y,alpha, axis=0)
    
    return ub, lb

def PINRW(real_y, generated_y, confidence=0.95):
    ub, lb=bound(generated_y, confidence)
    real_y=real_y.reshape(-1)
    R=max(real_y)-min(real_y)
    
    return np.sqrt(np.mean(pow(ub-lb,2)))/R
    

def PICP(real_y, generated_y, confidence=0.95, average=True):
    '''
    generated_y, should be shape of [num_experiments, num_sampling, N, dim], represents the generation results.
    real_y, should be shape of [N, dim].
    '''
    ub, lb=bound(generated_y, confidence)
    real_y=real_y.reshape(-1)
    b_lb=lb<=real_y
    b_ub=real_y<=ub
#    print(b_lb)
#    print(b_ub)
    b=(b_lb * b_ub)
    if average:
        return np.mean(b)
    else:
        return ub, lb, b_ub, b_lb

def PINAW(real_y, generated_y, confidence=0.95):    
    ub, lb=bound(generated_y, confidence)
    real_y=real_y.reshape(-1)
    R=max(real_y)-min(real_y)
    return np.mean(ub-lb)/R

def ACE(real_y, generated_y, confidence=0.95):
    return PICP(real_y, generated_y, confidence)-confidence

def CWC(real_y, generated_y, confidence=0.95, ita=50):
    ACE_score=ACE(real_y, generated_y, confidence)
    return PINAW(real_y, generated_y, confidence)+(ACE_score<0)*np.exp(-ita*(ACE_score))

def WS(real_y, generated_y, confidence=0.95):
    ub, lb, b_ub, b_lb=PICP(real_y, generated_y, confidence, average=False)
    real_y=real_y.reshape(-1)
    phi=-2*(1-confidence)*(ub-lb)-4*(1-b_lb)*(lb-real_y)-4*(1-b_ub)*(real_y-ub)
    return np.mean(phi)

def DIC(real_y, generated_y, confidence=0.95):
    ub, lb, b_ub, b_lb=PICP(real_y, generated_y, confidence, average=False)
    PINAW_score=PINAW(real_y, generated_y, confidence)
    ACE_score=ACE(real_y, generated_y, confidence)
    DIC_score=PINAW_score+(ACE_score<0)*1/(1-confidence)*np.sum((1-b_lb)*(lb-real_y)+(1-b_ub)*(real_y-ub))
    return DIC_score
    
    

if __name__=='__main__':
    real_y=np.zeros((3,6))
    generated_y=np.ones((2,3,3,6))*2
#    CRPS_score=CRPS(real_y,generated_y)
#    generated_y=np.ones((3,6))
    score=RMSE(real_y, generated_y, average=True)
    print(score)
