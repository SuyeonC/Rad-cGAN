#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pprint import pprint
from pysteps import io, nowcasts, rcparams, verification
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.postprocessing.ensemblestats import excprob
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field
from pysteps.postprocessing import ensemblestats


# In[2]:


#rain_inpu
def generate_data(data):
    n_samples=data.shape[0]
    time_step=4
    row=128
    col=128
    
    # for training dataset (t-30,t-20,t-10,t,1+10)
    movie_in=np.zeros((n_samples,time_step,row,col))
    movie_out=np.zeros((n_samples,1,row,col))
    for i in range(n_samples):
        for j in range(time_step):    
            m_in=pow(10.0,data[i,::,::,j]/10.0)
            movie_in[i,j,::,::]=pow(m_in/200.0,1.0/1.6)
        m_out=pow(10.0,data[i,::,::,-1]/10.0)
        movie_out[i,0,::,::]=pow(m_out/200.0,1.0/1.6)
    
    threshold=0.1
    zerovalue=-15
    zeros1=movie_in < threshold
    movie_in[~zeros1]=10*np.log10(movie_in[~zeros1])
    movie_in[zeros1]=zerovalue
    
    zeros2=movie_out < threshold
    movie_out[~zeros2]=10*np.log10(movie_out[~zeros2])
    movie_out[zeros2]=zerovalue

    
    return movie_in, movie_out


# In[3]:


def inverse_dB(data):

    threshold=-10
    zerovalue=0.0
    
    R=10.0**(data/10.0)
    threshold=10.0**(threshold/10.0)
    R[R<threshold]=zerovalue
    
    return R


# In[5]:


#### run for all samples ###
from IPython.display import clear_output
import datetime
path_result='/home/suyeonc/convlstm/rainfall/codes/results/pysteps_sprog/'
path_data='/home/suyeonc/rainpredict/data/train_data/'
a=[]
jj=27905
while True:
    try:
        for l in [8]:
            n_leadtimes=l
            timestep=10
            lead_time=l*timestep
            prediction=np.zeros((1,128,128))
            g_truth=np.zeros((1,128,128))
            

            for j in range(jj,len(dir_data)):
                
                path_test=path_data+dir_data[j] 
                after=dir_data[j][:-4]
                after_name=path_data+dir_data[j]
                after=datetime.datetime.strptime(after,'%Y%m%d%H%M')
                before=after-datetime.timedelta(minutes=lead_time-10)
                before=before.strftime('%Y%m%d%H%M')
                before_name=path_data+before+'.npy'
                if os.path.isfile(before_name)==True:
                    data_b=np.load(before_name)
                    data_a=np.load(after_name)
                    before_in,before_out=generate_data(data_b[np.newaxis,:])
                    after_in,after_out=generate_data(data_a[np.newaxis,:])       
        
                    R=before_in[0,:,:,:]
                    V = dense_lucaskanade(R)

            ## setting: default & example from manual
            #n_ens_members = 20 
            #seed = 24
    
            ## nowcast
                    nowcast_method = nowcasts.get_method("sprog")
                    R_f = nowcast_method(
                        R[:, :, :],
                        V,
                        n_leadtimes,
                        n_cascade_levels=6,
                        R_thr=-10.0,
                        )
                    clear_output(wait=True)
                    print(j)
    
            # Back-transform to rain rates
                    #R_f=inverse_dB(R_f)
                    #R_t=inverse_dB(after_out)
            
                    #prediction=np.append(prediction,R_f[np.newaxis,-1,:,:],axis=0)
                    #g_truth=np.append(g_truth,R_t[0,:,:,:],axis=0)
                                
    # save the ensemble mean
            #predict_out=prediction[1:,:,:]
            #true_out=g_truth[1:,:,:]
        
            #filename_p=path_result+'predict1_'+str(lead_time)+'min'
            #filename_o=path_result+'observ1_'+str(lead_time)+'min'
        
            #np.save(filename_p,predict_out)
            #np.save(filename_o,true_out)
            #print("saved!")
                
            
        break
        
        
    except:
        
        jj=j
        a=np.append(a,j+len(a))
        dir_data=np.delete(dir_data,j,axis=0)
        np.save(path_result+'removed_index_sprog1'+str(l)+'.npy',a)


# In[6]:


n_leadtimes=l
timestep=10
lead_time=l*timestep

data_b=np.load(path_data)
data_in,data_out=generate_data(data_b[np.newaxis,:])
   
        
R=data_in[0,:,:,:]
V = dense_lucaskanade(R)
            
## nowcast
nowcast_method = nowcasts.get_method("sprog")
R_f = nowcast_method(
        R[:, :, :],
        V,
        n_leadtimes,
        n_cascade_levels=6,
        R_thr=-10.0,
        )

R_f=inverse_dB(R_f)
R_t=inverse_dB(data_out)         

prediction=R_f[-1,:,:]
observation=R_t[0,:,:,:]

