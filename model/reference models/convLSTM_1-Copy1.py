#!/usr/bin/env python
# coding: utf-8

# In[20]:


from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv3D
from keras.layers import LayerNormalization
from keras.layers.normalization import BatchNormalization
model=Sequential()
model.add(ConvLSTM2D(filters=64,
                     kernel_size=(3,3),padding='same',
                     kernel_initializer='HeNormal',
                     input_shape=(None,128,128,1),
                     return_sequences=True))
model.add(LayerNormalization())
model.add(ConvLSTM2D(filters=64,
                     kernel_size=(3,3),padding='same',
                     kernel_initializer='HeNormal',
                     return_sequences=True))
model.add(LayerNormalization())
model.add(ConvLSTM2D(filters=64,
                     kernel_size=(3,3),padding='same',
                     kernel_initializer='HeNormal',
                     return_sequences=True))
model.add(LayerNormalization())
model.add(Conv3D(filters=1,
                 kernel_size=(3,3,3),padding='same',
                 activation='linear',
                 data_format='channels_last'))
model.compile(loss='mse',optimizer='adam')
model.summary()

