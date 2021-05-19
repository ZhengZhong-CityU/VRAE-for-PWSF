# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:16:25 2020

@author: zhongzheng
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from evaluations import PICP, bound, PINRW, PINAW, ACE, CWC, WS, DIC
import itertools
import pandas as pd
import os


Fontsize=20
sns.set_style("white")
np.random.seed(121)


def get_results(path='BeijingPM', metric='CRPS'):
    score_results={}
    prediction_results={}
    file_list=os.listdir(path)
    
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')
    
#    print(file_list)
    method_list=[]
    for file in file_list:
        method=file.split('_')[0]
        method_list.append(method)
#    method_list=list(set(method_list))
    


    print(method_list)
    val_dict={}
    for file in file_list:
        method=file.split('_')[0]
        new_metric=metric
        if method not in ['MVRAEGMM', 'GP', 'CKDE-NRC','DMDNN'] and metric=='CRPS':
            new_metric='MAE'
        if method in ['GRU', 'MVRAEGMM','DMDNN']:
            validation_root='validation_generation'
            test_root='test_generation'
            if method not in val_dict.keys():
                val_dict[method]=10000
            if method in ['DMDNN', 'DMDNNA']:
                validation_root=os.path.join('validation_generation', '500_2')
                test_root=os.path.join('test_generation', '500_2')
            if method=='MVRAEGMM':
                validation_root=os.path.join('validation_generation', '500_2_40_1')
                test_root=os.path.join('test_generation', '500_2_40_1')
                
            val_score_file='{}.txt'.format(new_metric)           
            val_score_path=os.path.join(path, file, validation_root, val_score_file)
#            print(val_score_path)
            with open(val_score_path, 'r') as f:
                current_val_score=float(f.read())
#                print(current_val_score)
        
            if current_val_score<val_dict[method]:
                val_dict[method]=current_val_score
                score_root=os.path.join(path, file, test_root)
            else:
                continue
    
            
        elif method in ['GP', 'CKDE-NRC']:
                score_root=os.path.join(path, file, 'test_generation', '500_2')
                
        elif method in ['ARIMA', 'nearest', 'Linear']:
                score_root=os.path.join(path, file, 'test_generation')
        else:
            continue
                
        
        pred_file='generated_y.npy'   
        real_file='real_y.npy'
        score_file='{}.txt'.format(new_metric)
        
        if method=='DT':
            method='Decision tree'
        if method=='MDNGMMGRU':
            method='MDN-GMM'
        if method=='MVRAEGMM':
            method='VRAE-GMM'
        if method=='nearest':
            method='PC'
        if method=='GP':
            method='GPR'
    
        score_file_path=os.path.join(score_root, score_file)
        with open(score_file_path, 'r') as f:  
    #        print(file_path, f.read())
            score_results[method]=float(f.read())
            
        pred_file_path=os.path.join(score_root, pred_file)
        prediction_results[method]=np.load(pred_file_path)
        if prediction_results[method].ndim==1:
            prediction_results[method]=np.expand_dims(prediction_results[method],-1)
        if prediction_results[method].ndim==3 or prediction_results[method].ndim==5 :
            prediction_results[method]=np.squeeze(prediction_results[method],-1)
        
        if 'Observation' not in prediction_results.keys():
            real_file_path=os.path.join(score_root, real_file)
            prediction_results['Observation']=np.load(real_file_path)
            if prediction_results['Observation'].ndim==1:
                prediction_results['Observation']=np.expand_dims(prediction_results['Observation'],-1)
    return score_results, prediction_results
                


#RMSE_score_results, RMSE_prediction_results=get_results(path='BeijingPM', metric='RMSE')
    

def get_prediction_results(path='AE'):
    result={}
    file_list=os.listdir(path)
    for file in file_list:
        method=file.split('_')[0]
        generated_file='generated_y.npy'
        root='test_generation'
        if method in ['GP', 'CKDE-NRC', 'DMDNN']:
            root=root+'/500_2'
        if method=='MVRAEGMM':
            root=root+'/500_2_40_1' 
        if 'Ground truth' not in result.keys():
            real_y=np.load(os.path.join(path, file,root, 'real_y.npy'))
            if real_y.ndim==1:
                real_y=np.expand_dims(real_y,-1)
            result['Ground truth']=real_y    
        generated_y=np.load(os.path.join(path, file,root, generated_file))
        if generated_y.ndim==1:
            generated_y=np.expand_dims(generated_y,-1)
        if method=='DT':
            method='Decision Tree'
        if method=='MDNGMMGRU':
            method='MDN-GMM'
        if method=='MVRAEGMM':
            method='VRAE-GMM'
        if method=='nearest':
            method='PC'
        if method=='GP':
            method='GPR'
#        print(method)
        result[method]=generated_y
    return result


def plot_samples(dataset, result, num_images=5):
    file=os.path.join('Figures', dataset)
    if not os.path.exists(file):
        os.makedirs(file)
        
    num_images=num_images
    num_samples= result['Observation'].shape[0]  
    
    random_indices=np.random.choice(num_samples, num_images)
    for i in range(num_images):
        fig=plt.figure(figsize=((16,8)))
        for C, method in enumerate(result.keys()):
            if method in ['VRAE-GMM', 'GPR', 'CKDE-NRC','DMDNN']:
                data=result[method][0,:,random_indices[i],0]
                sns.kdeplot(data, shade = True, label=method, color='C{}'.format(C))
#                sns.distplot(data, label=method, color='C{}'.format(C), kde=True, norm_hist=True, hist=False)
            if method in ['ARIMA', 'GRU', 'Linear', 'PC']:
                data=result[method][random_indices[i],0]
                plt.axvline(data, label=method, color='C{}'.format(C))
            if method=='Observation':
                data=result[method][random_indices[i],0]
                plt.axvline(data, label=method, linestyle='--', marker='*', lw=4, color='C{}'.format(C))
        
        plt.xlabel('Wind speed (m/s)',fontsize=Fontsize)
        plt.ylabel('Density',fontsize=Fontsize)                
        plt.legend(fontsize=Fontsize, facecolor='white')
        plt.xticks(fontsize=Fontsize)
        plt.yticks(fontsize=Fontsize)

        
        path=os.path.join(file,'samples_{}.png'.format(i))
        fig.savefig(path, dpi=600, bbox_inches='tight')

def plot_predictions(dataset, result, ub_results, lb_results, start_point=0, end_point=200):
    
    file=os.path.join('Figures', dataset)
    if not os.path.exists(file):
        os.makedirs(file)
    
    start_point=start_point
    end_point=end_point
    
    # figure 1
    fig1=plt.figure(figsize=((16,8)))
    for C, method in enumerate(result.keys()):
        data=result[method][start_point:end_point,0]
        if method in ['VRAE-GMM', 'GPR', 'CKDE-NRC','DMDNN']:
            pass
#            data=result[method][0,0,start_point:end_point,0]
#            plt.plot(data, label=method+' random sample',linestyle='--', color='C{}'.format(C))
        elif method=='Observation':
            plt.plot(data, label=method, marker='*',color='C{}'.format(C))
        else:
            plt.plot(data, label=method, color='C{}'.format(C))
    plt.xlabel('Time (hour)',fontsize=Fontsize)
    plt.ylabel('Wind speed (m/s)',fontsize=Fontsize)
    plt.legend(fontsize=Fontsize, facecolor='white')
    plt.xticks(fontsize=Fontsize)
    plt.yticks(fontsize=Fontsize)
    path1=os.path.join(file, 'deterministic_predictions.png')
    fig1.savefig(path1, dpi=600, bbox_inches='tight')
     
    # figure 2
    fig2=plt.figure(figsize=((16,8)))
    for C, method in enumerate(result.keys()):
        if method in ['VRAE-GMM', 'GPR', 'CKDE-NRC','DMDNN']:
            ub=ub_results[method][start_point:end_point]
            lb=lb_results[method][start_point:end_point]
#            data=result[method][0,0,start_point:end_point,0]
#            plt.plot(data, label=method+' random sample',linestyle='--', color='C{}'.format(C))
            plt.plot(ub, label=method+ ' upper bound', marker='o', color='C{}'.format(C))
            plt.plot(lb, label=method+ ' lower bound', marker='v', color='C{}'.format(C))
        elif method=='Observation':
            data=result[method][start_point:end_point,0]
            plt.plot(data, label=method, marker='*',color='C{}'.format(C))
        else:
            pass
    plt.xlabel('Time (hour)',fontsize=Fontsize)
    plt.ylabel('Wind speed (m/s)',fontsize=Fontsize)
    plt.legend(fontsize=Fontsize, facecolor='white')
    plt.xticks(fontsize=Fontsize)
    plt.yticks(fontsize=Fontsize)
    path2=os.path.join(file,'probabilistic_predictions.png')
    fig2.savefig(path2, dpi=600, bbox_inches='tight')


def performance(dataset='ShenyangPM', confidence=0.95):
    confidence=confidence
    CRPS_score_results, CRPS_prediction_results=get_results(path=dataset, metric='CRPS')

    PICP_score_results={}
    PINRW_score_results={}
    PINAW_score_results={}
    ACE_score_results={}
    CWC_score_results={}
    WS_score_results={}
    DIC_score_results={}

    ub_results={}
    lb_results={}

    for method in ['VRAE-GMM', 'GPR','CKDE-NRC', 'DMDNN']:
        generated_y=CRPS_prediction_results[method]
        real_y=CRPS_prediction_results['Observation']
        ub, lb=bound(generated_y, confidence)
        ub_results[method]=ub
        lb_results[method]=lb
        PICP_score=PICP(real_y, generated_y, confidence)
        PINRW_score=PINRW(real_y, generated_y, confidence)
        PINAW_score=PINAW(real_y, generated_y, confidence)
        ACE_score=ACE(real_y, generated_y, confidence)
        CWC_score=CWC(real_y, generated_y, confidence)
        WS_score=WS(real_y, generated_y, confidence)
        DIC_score=DIC(real_y, generated_y, confidence)
        PICP_score_results[method]=PICP_score
        PINRW_score_results[method]=PINRW_score
        PINAW_score_results[method]=PINAW_score
        ACE_score_results[method]=ACE_score
        CWC_score_results[method]=CWC_score
        WS_score_results[method]=WS_score
        DIC_score_results[method]=DIC_score
    
    print(dataset)
    print(confidence)
    print('PINC: {}' .format(confidence))
    print('PICP: {}'.format(PICP_score_results))
    print('PINRW: {}'.format(PINRW_score_results))
    print('PINAW: {}'.format(PINAW_score_results))
    print('ACE: {}'.format(ACE_score_results))
    print('CWC: {}'.format(CWC_score_results))
    print('WS: {}'.format(WS_score_results))
    print('DIC: {}'.format(DIC_score_results))
    
    
    return CRPS_score_results, PICP_score_results,PINRW_score_results,PINAW_score_results,ACE_score_results,CWC_score_results,WS_score_results,DIC_score_results


    
def plot_figures(dataset='ShenyangPM', confidence=0.95):
    CRPS_score_results, CRPS_prediction_results=get_results(path=dataset, metric='CRPS')
    ub_results={}
    lb_results={}
    for method in ['VRAE-GMM', 'GPR','CKDE-NRC', 'DMDNN']:
        generated_y=CRPS_prediction_results[method]
        ub, lb=bound(generated_y, confidence)
        ub, lb=bound(generated_y, confidence)
        ub_results[method]=ub
        lb_results[method]=lb
    plot_samples(dataset, CRPS_prediction_results, 5)
    plot_predictions(dataset, CRPS_prediction_results, ub_results, lb_results, 0, 50)
    

def summarize(Prob_indices, dataset):
 
    data_dict=Prob_indices[dataset]

    confidence_list=list(Prob_indices[dataset].keys())
    method_list=['GPR', 'CKDE-NRC', 'DMDNN', 'VRAE-GMM']
    
    result_final=[]
    for method, confidence in itertools.product(method_list, confidence_list):
        result_one_row=[]
        data_list=data_dict[confidence]
        for i in range(len(data_list)):
            item=data_list[i][method]
            result_one_row.append(item)
        result_one_row=np.asarray(result_one_row)
        result_final.append(result_one_row)
    result_final=np.asarray(result_final)
    

    save_path='../excel_results_210419'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_file=os.path.join(save_path, '{}.xlsx'.format(dataset))
    df=pd.DataFrame (result_final)
    df.index=['{}-{}'.format(i[0],i[1]) for i in itertools.product(method_list,confidence_list)]
 
    df.columns=['PICP','ACE','PINRW','PINAW','CWC','WS']

    df.to_excel(save_file)
    
    print('excel saved to {}'.format(save_file))
    
    return result_final  

def save_CRPS(CRPS_dict):
    dataset_list=list(CRPS_dict.keys())
    method_list=['PC', 'ARIMA','Linear','GRU', 'GPR','CKDE-NRC', 'DMDNN', 'VRAE-GMM']
    final_result=[]
    for dataset in dataset_list:
        result_oneset=[]
        for method in method_list:
            item=CRPS_dict[dataset][method]
            result_oneset.append(item)
        result_oneset=np.asarray(result_oneset)
        final_result.append(result_oneset)
    final_result=np.asarray(final_result)
    
    save_path='../excel_results_210419'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_file=os.path.join(save_path, '{}.xlsx'.format('CRPS'))
    df=pd.DataFrame (final_result)
    df.index=dataset_list
 
    df.columns=method_list

    df.to_excel(save_file)
    
    print('excel saved to {}'.format(save_file))
    
    return final_result 
    
    
    


    
if __name__=='__main__':

    dataset_list=['BeijingPM', 'ChengduPM', 'ShanghaiPM', 'GuangzhouPM', 'ShenyangPM']
#    dataset_list=['BeijingPM']
#   
    
    ## get results
    CRPS_dict={}
    Prob_indices={}
    confidence_list=[0.9, 0.95] 
    
    
    for dataset, confidence in itertools.product(dataset_list, confidence_list):
        if dataset not in Prob_indices.keys():
            Prob_indices[dataset]={}
        CRPS_score_results, PICP_score_results,PINRW_score_results,PINAW_score_results,ACE_score_results,CWC_score_results,WS_score_results,DIC_score_results=performance(dataset, confidence)
        CRPS_dict[dataset]=CRPS_score_results
        Prob_indices[dataset][confidence]=[PICP_score_results,ACE_score_results,PINRW_score_results,PINAW_score_results,CWC_score_results,WS_score_results]
#    
    ## plot figures    
    for dataset in dataset_list:
        plot_figures(dataset, 0.6)

    # save probabilistic evaluation results to excel        
    for dataset in dataset_list:       
        result_final=summarize(Prob_indices, dataset)
    
    # save probabilistic evaluation results to excel     
    save_CRPS(CRPS_dict)

        
        
    
        
       
             







    








    