#! /usr/bin/env python
# -*- coding: utf-8 -*-
import conv3dmodels as MD
from keras.utils import np_utils
from keras.models import model_from_json                                         
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
import h5py
import os                                                                        
import cv2                                                                       
import numpy as np                                                                                                           
import matplotlib.pyplot as plt
import sys                                                                       
import keras.backend as K

gpu_id='7'
os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

dim_ordering = K._image_dim_ordering
print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering)      
backend = dim_ordering

import tensorflow as tf
tf.python.control_flow_ops=tf
####################################load weights by name###################################
tune = 0
model_dir = './models' 
model_weights_name = 'sports1M_weights_tf.h5'
model_weight_filename = os.path.join(model_dir, model_weights_name)
model = MD.generate_model(backend=backend, decomid=int(tune))
model.load_weights(model_weight_filename, by_name=True)

#for layer in model.layers[0:19]:
#    layer.trainable = False
model.summary()
#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',
#              optimizer=SGD(lr=1e-4, momentum=0.9),
#              metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
####################################load ucf for finetune#####################################
nb_epoch = 11112
batch_size = 500
def data_process(X, y, nb_class):
    X  = X.astype('float32')
    X /= 255
    X -= np.mean(X)
    y  = np_utils.to_categorical(y, nb_class)
    return X, y
def generate_sequence(image_path):
    fp = open(image_path, 'r')
    file = fp.readlines()
    nb_samples = len(file)
    sequence = np.random.permutation(nb_samples)
    fp.close()
    return sequence
def load_train_data(select):
    trainfile = h5py.File('/home/yutingzhao/VideoData/UCF-h5/ucf16frame-train3.h5')
    X_train = trainfile['X_train'][select,:,:,:,:]
    y_train = trainfile['y_train'][select,]
    return X_train, y_train
sequence = generate_sequence('/home/yutingzhao/VideoData/UCF-RGB/trainlist03.txt')
nb_samples = len(sequence)
def load_test_data(select):
    testfile = h5py.File('/home/yutingzhao/VideoData/UCF-h5/ucf16frame-test3.h5')
    X_test = testfile['X_test'][select,:,:,:,:]
    y_test = testfile['y_test'][select,]
    return X_test, y_test
test_sequence = generate_sequence('/home/yutingzhao/VideoData/UCF-RGB/testlist03.txt')
nb_test_samples = len(test_sequence)

for ep in range(nb_epoch):
    times = range(int(nb_samples / batch_size))#9624/100=96
    for ind in times:
	select = np.sort(sequence[ind * batch_size:(ind + 1) * batch_size])
        select = select.tolist()
        X_batch, y_batch = load_train_data(select)
	X_batch, y_batch = data_process(X_batch, y_batch, 101)
	X_batch = np.rollaxis(X_batch, 3, 1)#fit sport1mformat
	print X_batch.shape, y_batch.shape
        print('epoch %d/%d, ind %d/%d' % (ep + 1, nb_epoch, ind + 1, int(nb_samples / batch_size)))
# demand shape (None, 16, 112, 112, 3)
	X_batch = X_batch[:,:,30:142, 8:120, :]
        model.fit(X_batch, y_batch, batch_size=5, nb_epoch=15, verbose=1)
#################predict ucf test##################
	if ind%18==0 and ind>0 or ind==len(times)-1:
	#if ind>=0:
            print ind
	    test_select = np.sort(test_sequence[0:100])
            test_select = test_select.tolist()
            X_test, y_test = load_test_data(test_select)
            X_test, y_test = data_process(X_test, y_test, 101)
            X_test = np.rollaxis(X_test, 3, 1)#fit sport1mformat
            print X_test.shape, y_test.shape
            X_test = X_test[:,:,30:142, 8:120, :]
            output = model.predict(X_test, batch_size=5, verbose=1)
            output = np.argmax(output, axis=1)
            y_test = np.argmax(y_test, axis=1)
            print("Accuracy: ", 1.*np.sum(output==y_test)/y_test.shape[0])
#################end for test#####################
#    modelbasename = 'c3d_ucf101_finetune_onsport1m_'
#    model.save(model_dir+modelbasename+'tune'+str(tune)+'Ep'+str(ep)+'.h5')
########################################end for finetune######################################
