#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:06:57 2021

@author: zhangyan
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def torch_gamma_rp(a, b):
    ## a is alpha>0
    ## b is beta=1/scale >0
    ## support (0, +inf)
    # print(a)
    x=Variable(torch.randn(a.shape)).to(device) 

    z=torch.exp(torch.sqrt(1/a+1/2/torch.pow(a,2))*x-torch.log(b)+torch.log(a)-1/(2*a))
    return z


def torch_beta_rp(a, b):
    ## a is alpha>0
    ## b is beta=1/scale >0
    ## support [0,1]
    z1=torch_gamma_rp(a, torch.ones(a.shape).to(device))
    z2=torch_gamma_rp(b, torch.ones(b.shape).to(device))
    z=z1/(z1+z2)
    return z


if __name__=='__main__':
#    size=1000
    a=torch.tensor([[1.0,2.0],[3.0,4.0]])
    b=torch.tensor([[0.5,0.5],[0.5,0.5]])
    z=torch_beta_rp(a, b)
    print(z)
#    fig=plt.figure()
#    sns.kdeplot(z.numpy(), c='r', label='alpha: {} beta: {} fake'.format(a, b))
    m=torch.distributions.beta.Beta(a,b)
    
    samples=m.sample()

    
