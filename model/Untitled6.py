#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def generate_data(data):
    n_samples=data.shape[0]
    time_step=data.shape[3]
    row=128
    col=128
    
    # for training dataset (t-30,t-20,t-10,t,1+10)   
    n_frames=4
    movie_in=np.zeros((n_samples,row,col,n_frames))
    movie_out=np.zeros((n_samples,row,col,1))
    for i in range(n_samples):
        for j in range(n_frames):
            m_in=(255.*((data[i,::,::,j]+10.)/70.))+0.5
            movie_in[i,::,::,j]=m_in
        m_out=(255.*((data[i,::,::,-1]+10.)/70.))+0.5
        movie_out[i,::,::,0]=m_out

    return movie_in, movie_out

