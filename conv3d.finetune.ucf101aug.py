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
print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(dim_ordering)
backend = dim_ordering

import tensorflow as tf
tf.python.control_flow_ops=tf

tune = 0
model_dir = './models'
model_weights_name = 'sports1M_weights_tf.h5'
model_weight_filename = os.path.join(model_dir, model_weights_name)
model = MD.generate_model(backend=backend, decomid=int(tune))
model.load_weights(model_weight_filename, by_name=True)
#model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy']) 

nb_epoch = 5000
batch_size = 200
nb_classes = 101

def data_process(X, y, nb_class):
    X  = X.astype('float32')
    X /= 255
    X -= np.mean(X)
    y  = np_utils.to_categorical(y, nb_class)
    return X, y
def generate_sequence(split):
    with open('/home/yutingzhao/VideoData/UCF-h5/UCF-Augmentation-224/trainsplit'+str(split)+'.txt', 'r') as fp:
        lines = fp.readlines()
        train_samples = len(lines)
        sequence = np.random.permutation(train_samples)
        fp.close()
    return sequence
def load_test_data(select):
    testfile = h5py.File('/home/yutingzhao/VideoData/UCF-h5/ucf16frame-test3.h5')
    X_test = testfile['X_test'][select,:,:,:,:]
    y_test = testfile['y_test'][select,]
    return X_test, y_test
def generate_test_sequence(image_path):
    fp = open(image_path, 'r')
    file = fp.readlines()
    nb_samples = len(file)
    sequence = np.random.permutation(nb_samples)
    fp.close()
    return sequence

test_sequence = generate_test_sequence('/home/yutingzhao/VideoData/UCF-RGB/testlist03.txt')
nb_test_samples = len(test_sequence)

def load_train_data(sequence):
    train_path  = '/home/yutingzhao/VideoData/UCF-h5/UCF-Augmentation-224/UCF-224-part'
    X_train     = []
    y_train     = []
    for select in sequence:
        part    = int(select) // 10000 + 1
        index   = int(select) % 10000
        train   = h5py.File(train_path+str(part)+'.h5')
        X_data  = train['X_train'][index,:,:,:,:]
        y_data  = train['y_train'][index]
        X_train.append(X_data)
        y_train.append(y_data)
    X_train     = np.asarray(X_train)
    y_train     = np.asarray(y_train)
    return X_train, y_train

train_split = 2
sequence = generate_sequence(train_split)
nb_iteration= len(sequence) // batch_size 


accuracy_epoch = []
for ep in range(nb_epoch):
    accuracy_iteration = []
    for it in range(nb_iteration):
        select_sequence = sequence[it*batch_size:(it+1)*batch_size]
        X_train, y_train= load_train_data(select_sequence)
        X_train, y_train= data_process(X_train, y_train, nb_classes)
        X_train = np.rollaxis(X_train, 3, 1)#fit sport1mformat
        X_train = X_train[:,:,30:142, 8:120, :]
        print('epoch %d/%d, iteration %d/%d' % (ep+1, nb_epoch, it+1, nb_iteration))
        #model.fit(X_train, y_train, batch_size=5, nb_epoch=10, verbose=1, validation_split=0.2)
        model.fit(X_train, y_train, batch_size=5, nb_epoch=10, verbose=1)
	if it%10==0 and it>0 or it==nb_iteration-1:
        #if it>=0:
            print it
            print ("#########################test data shape#############################")
            test_select = np.sort(test_sequence[0:500])
            test_select = test_select.tolist()
            X_test, y_test = load_test_data(test_select)
            X_test, y_test = data_process(X_test, y_test, 101)
            X_test = np.rollaxis(X_test, 3, 1)#fit sport1mformat
            X_test = X_test[:,:,30:142, 8:120, :]
	    print X_test.shape, y_test.shape
            print ("###########################end for test shape########################")
            output = model.predict(X_test, batch_size=5, verbose=1)
            output = np.argmax(output, axis=1)
            y_test = np.argmax(y_test, axis=1)
            acc =  1.*np.sum(output==y_test)/y_test.shape[0]
            print("Accuracy: ", acc)
            accuracy_iteration.append(acc)
            print("Iteration Accuracy: ", accuracy_iteration)
        if it%100==0 and it>0 or it==nb_iteration-1:
	    modelbasename = 'c3d_ucf101_finetune_onsport1m_'
	    model.save(model_dir+modelbasename+'tune'+str(tune)+'Ep'+str(ep)+'iter'+str(it+1)+'.h5')
            
    accuracy_epoch.append(accuracy_iteration)
    print("Epoch Accuracy: ", accuracy_epoch)
    print("max accuracy: ", max(max(accuracy_epoch)))

