#!/usr/bin/env python
# coding: utf-8


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

# generate input data(unit:dBR) converted from raw data(dBZ)
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


# calculate rain rate R(mm/hr) from dBR
def inverse_dB(data):

    threshold=-10
    zerovalue=0.0
    
    R=10.0**(data/10.0)
    threshold=10.0**(threshold/10.0)
    R[R<threshold]=zerovalue
    
    return R


#### run for samples ###
path_data='path of data downloaded'

n_leadtimes=9 # predict for lead time of 90 min
timestep=10
lead_time=n_leadtimes*timestep

# define empty arrays for prediction and observation
prediction=np.zeros((1,128,128))
observation=np.zeros((1,128,128))


raw_data=np.load(path_data)
data_in,data_out=generate_data(raw_data[np.newaxis,:])
   
for i in range(len(raw_data)):        
    R=data_in[i,:,:,:]
    V = dense_lucaskanade(R)
            
    # nowcast
    nowcast_method = nowcasts.get_method("sprog")
    R_f = nowcast_method(
            R[:, :, :],
            V,
            n_leadtimes,
            n_cascade_levels=6,
            R_thr=-10.0,
            )
    
    # transform dBR results to R(mm/hr)
    R_f=inverse_dB(R_f)
    R_t=inverse_dB(data_out[i])  
    
    # add to arrays for prediction and observation
    prediction=np.append(prediction,R_f[np.newaxis,-1,:,:],axis=0)
    observation=np.append(observation,R_t[0,:,:,:],axis=0)
   
# remove first empty index from arrays for prediction and observation
prediction=predictino[1:,:,:]
observation=observation[1:,:,:]

