import os
import math
import pickle
import numpy as np
import configparser
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, LeakyReLU,PReLU

from helpers import load_hdf5

#-----------------------------------------------------------------------------
# Exponential Decay of learning rate

def exp_decay(epoch):
   initial_lrate = 0.2
   k = 0.1
   lrate = initial_lrate * math.exp(-k*epoch)
   return lrate

#-----------------------------------------------------------------------------

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    
    shortcut = y

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', use_bias=False)(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(y)

    if _project_shortcut or _strides != (1, 1):
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', use_bias=False)(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

#-----------------------------------------------------------------------------
# define residual model
    
def getSampleResidualModel2(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
    act1 = LeakyReLU(alpha=0.1)(conv1)
    res1 = residual_block(act1, 64);  
    pool1 = MaxPool2D(pool_size=(2,2))(res1)
    
    res2 = residual_block(pool1, 64);  
    pool2 = MaxPool2D(pool_size=(2,2))(res2)
        
    res3 = residual_block(pool2, 64)
    pool3 = MaxPool2D(pool_size=(2,2))(res3)
    
    res4 = residual_block(pool3, 64)
    pool4 = MaxPool2D(pool_size=(2,2))(res4)
    
    flat1 = Flatten()(pool4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)

    initial_learning_rate = 0.001
    decay_rate = 0.001

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=decay_rate,
    staircase=True
    )

    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model

#-----------------------------------------------------------------------------
#read config file

config = configparser.RawConfigParser()
config.read('configuration.txt')

num_epochs = int(config.get('training settings', 'num_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))


num_classes = int(config.get('data attributes','num_classes'))
patch_size = int(config.get('data attributes','patch_size'))

train_patches = config.get('data paths','train_patches')
train_labels = config.get('data paths','train_labels')
best_weights = config.get('data paths','best_weights')

history_saved = config.get('data paths','history_saved')

augment = config.get('training settings', 'augment')

#-----------------------------------------------------------------------------
#load and transform data
os.environ["CUDA_VISIBLE_DEVICES"]="1"

X_train = load_hdf5(train_patches)
Y_train = load_hdf5(train_labels)
Y_train = to_categorical(Y_train)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

#get model and train it
model = getSampleResidualModel2(num_classes, (patch_size, patch_size, 3))

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    
checkpointer = ModelCheckpoint(best_weights, verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
patienceCallBack = EarlyStopping(monitor='val_loss',patience=50)
learningRateCallBack = LearningRateScheduler(exp_decay ,verbose = 1)
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)

batch_size = int(batch_size)


if augment == 'True':
    
    print('Train with augmentation')
    
    datagen = ImageDataGenerator(
                            rotation_range=10,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')
    
    validation_split = 0.2
    indx = list(range(0,len(X_train)))
    random.shuffle(indx)
    random.shuffle(indx)

    k = int(0.2*len(X_train))

    X_val = X_train[indx[0:k]]
    Y_val = Y_train[indx[0:k]]

    X_train = X_train[indx[k:len(X_train)]]
    Y_train = Y_train[indx[k:len(Y_train)]]


    history = model.fit(datagen.flow(x = X_train, y = Y_train, batch_size = batch_size),
                                  #validation_data = datagen.flow(x = X_val, y = Y_val, batch_size = batch_size),
                                  validation_data = (X_val, Y_val),
                                  steps_per_epoch = len(X_train)/batch_size,
                                  epochs = num_epochs,
                                  #callbacks = [checkpointer,tbCallBack,patienceCallBack])                  
                                  callbacks = [checkpointer,patienceCallBack])        
else:
    
    print('No augmentation')
    
    history = model.fit(x=X_train, 
                    y=Y_train, 
                    validation_split=0.2, 
                    epochs = num_epochs, 
                    batch_size=batch_size, 
                    shuffle=True, # shuffle: 10 l?p 7,6,6
                    callbacks = [checkpointer,tbCallBack,patienceCallBack])

#-----------------------------------------------------------------------------
#plot train history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc',color = "r")
plt.plot(epochs, val_acc, 'b', label='Validation acc',color = "magenta")
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss',color = "green")
plt.plot(epochs, val_loss, 'b', label='Validation loss',color = "cyan")
plt.title('Training and validation loss')
plt.legend()
plt.show()

pickle.dump(history, open(history_saved, "wb"))