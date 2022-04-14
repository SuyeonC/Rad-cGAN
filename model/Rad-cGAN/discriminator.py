#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras import Sequential
import numpy as np

def discriminate_model(lr,ps):
    in_inputs=Input(shape=(128,128,4),name='input')
    tar_inputs=Input(shape=(128,128,1),name='target')
    
    concat=concatenate([in_inputs,tar_inputs],axis=3)
    
    conv1 = Conv2D(64, 4,strides=(2,2), padding='same', kernel_initializer='he_normal')(concat)
    bn1 = BatchNormalization()(conv1)
    act1=LeakyReLU(alpha=0.2)(bn1)
    
    conv2 = Conv2D(128, 4,strides=(2,2), padding='same', kernel_initializer='he_normal')(act1)
    bn2 = BatchNormalization()(conv2)
    act2=LeakyReLU(alpha=0.2)(bn2)
    
    conv4 = Conv2D(256, 4,strides=(1,1), padding='same', kernel_initializer='he_normal')(act2)
    bn4 = BatchNormalization()(conv4)
    act4=LeakyReLU(alpha=0.2)(bn4)

    conv=Conv2D(1,4,padding='same',kernel_initializer='he_normal')(act4)
    outputs=Activation('sigmoid')(conv)
    
    model=Model([in_inputs,tar_inputs],outputs)
    
    opt=Adam(lr=lr,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt,loss_weights=[0.5])
    return model

