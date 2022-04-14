#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

def u_net_model(input_shape=(128,128,4)):
    
    inputs = Input(input_shape)


    conv1s = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(inputs)
    bn1s = BatchNormalization()(conv1s)
    act1s=Activation('relu')(bn1s)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act1s)
    drop1=Dropout(0.5)(pool1)

    conv2f = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop1)
    bn2f = BatchNormalization()(conv2f)
    act2f=Activation('relu')(bn2f)
    conv2s = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(act2f)
    bn2s = BatchNormalization()(conv2s)
    act2s=Activation('relu')(bn2s)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act2s)
    drop2=Dropout(0.5)(pool2)

    conv3f = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(drop2)
    bn3f = BatchNormalization()(conv3f)
    act3f=Activation('relu')(bn3f)
    drop3 = Dropout(0.5)(act3f)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(drop3), act2s], axis=3)
    conv4f = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(up4)
    bn4f = BatchNormalization()(conv4f)
    act4f=Activation('relu')(bn4f)
    drop4f=Dropout(0.5)(act4f)
    conv4 = Conv2D(256, 3, padding='same',activation='relu', kernel_initializer='he_normal')(drop4f)
    bn4 = BatchNormalization()(conv4)
    act4=Activation('relu')(bn4)
    
    up5 = concatenate([UpSampling2D(size=(2, 2))(act4), act1s], axis=3)
    conv5f = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up5)
    bn5f = BatchNormalization()(conv5f)
    act5f=Activation('relu')(bn5f)
    drop5f=Dropout(0.5)(act5f)
    conv5s = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop5f)
    bn5s = BatchNormalization()(conv5s)
    act5s=Activation('relu')(bn5s)
    drop5s=Dropout(0.5)(act5s)
    conv5 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(drop5s)
    bn5 = BatchNormalization()(conv5)
    act5=Activation('relu')(bn5)
    
    outputs = Conv2D(1, 1, activation='linear')(act5)

    model = Model(inputs=inputs, outputs=outputs)
    return model

