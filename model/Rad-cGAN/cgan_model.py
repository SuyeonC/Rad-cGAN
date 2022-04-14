#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras import Sequential
import numpy as np

def gan_model(lr,generator,discriminator):
    discriminator.trainable=False
    
    inputs=Input(shape=(128,128,4))
    gen_out=generator(inputs)
    dis_out=discriminator([inputs,gen_out])
    
    model=Model(inputs,[dis_out,gen_out])
    
    opt=Adam(lr=ir,beta_1=0.5)
    model.compile(loss=['binary_crossentropy','mae'],optimizer=opt,loss_weights=[1,100])
    return model

