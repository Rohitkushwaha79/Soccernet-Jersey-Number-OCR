
import tensorflow as tf
import numpy as np


from tensorflow import keras

from keras.layers import (GlobalAveragePooling2D, Conv2D, MaxPool2D, Dense,
                          Flatten, InputLayer, BatchNormalization, Input,
                          Dropout,Bidirectional, LSTM ,GRU, MaxPooling2D , Reshape, RandomRotation,RandomContrast,RandomBrightness)

from keras import backend as K
from keras.optimizers import Adam , SGD 
from keras.callbacks import  ReduceLROnPlateau 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Lambda
from layer.ctc_layer import CTCLayer


from keras import backend as k
from keras.preprocessing.image import img_to_array, load_img 

char_list =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_model():
    inputs = Input(shape=(32, 64, 1), name="image")


    labels = Input(name="label", shape=(None,), dtype="float32")


    conv_1 = Conv2D(32, (3,3), kernel_initializer="he_uniform" ,activation = "selu", padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(64, (3,3), activation = "selu", padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)


    conv_3 = Conv2D(128, (3,3), activation = "selu", padding='same')(pool_2)
    conv_4 = Conv2D(128, (3,3), activation = "selu", padding='same')(conv_3)


    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(256, (3,3), activation = "selu", padding='same')(pool_4)


    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(256, (3,3), activation = "selu", padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(64, (2,2), activation = "selu")(pool_6)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(GRU(128, return_sequences=True , dropout=0.3))(squeezed)
    blstm_2 = Bidirectional(GRU(128, return_sequences=True , dropout=0.3))(blstm_1)


    softmax_output = Dense(len(char_list) + 1, activation = 'softmax', name="dense")(blstm_2)


    output = CTCLayer(name="ctc_loss")(labels, softmax_output)

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1.0)


    #model to be used at training time
    model = keras.models.Model(inputs=[inputs, labels], outputs=output)
    model.compile(optimizer = optimizer)



    return model

