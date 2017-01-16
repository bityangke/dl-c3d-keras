from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD

# 'th'dim_ordering: [batch, channels, depth, height, width]
# 'tf'dim_ordering: [batch, depth, height, width, channels]
def generate_model(summary=True, backend='tf', decomid=0, nb_classes=101):
    model = Sequential()
    if backend == 'tf':
        input_shape=(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(3, 16, 112, 112) # c, l, h, w

    if decomid==0 :
        # 1st layer group
        model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv2'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a'))
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b'))
        model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
    elif decomid==1 :
        # 1st layer group
        model.add(Convolution3D(64, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv1_1',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv1_2',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv1_3',
                                input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv2'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a'))
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b'))
        model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
    elif decomid==2 :
        # 1st layer group
        model.add(Convolution3D(64, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv1_1',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv1_2',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv1_3',
                                input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv2_1'))
        model.add(Convolution3D(128, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv2_2'))
        model.add(Convolution3D(128, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv2_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a'))
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b'))
        model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
    elif decomid==3 :
        # 1st layer group
        model.add(Convolution3D(64, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv1_1',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv1_2',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv1_3',
                                input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv2_1'))
        model.add(Convolution3D(128, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv2_2'))
        model.add(Convolution3D(128, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv2_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv3a_1'))
        model.add(Convolution3D(256, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv3a_2'))
        model.add(Convolution3D(256, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv3a_3'))
        model.add(Convolution3D(256, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv3b_1'))
        model.add(Convolution3D(256, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv3b_2'))
        model.add(Convolution3D(256, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv3b_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b'))
        model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
    elif decomid==4 :
        # 1st layer group
        model.add(Convolution3D(64, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv1_1',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv1_2',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv1_3',
                                input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv2_1'))
        model.add(Convolution3D(128, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv2_2'))
        model.add(Convolution3D(128, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv2_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv3a_1'))
        model.add(Convolution3D(256, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv3a_2'))
        model.add(Convolution3D(256, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv3a_3'))
        model.add(Convolution3D(256, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv3b_1'))
        model.add(Convolution3D(256, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv3b_2'))
        model.add(Convolution3D(256, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv3b_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv4a_1'))
        model.add(Convolution3D(512, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv4a_2'))
        model.add(Convolution3D(512, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv4a_3'))
        model.add(Convolution3D(512, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv4b_1'))
        model.add(Convolution3D(512, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv4b_2'))
        model.add(Convolution3D(512, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv4b_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a'))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b'))
        model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        #elif decomid==5 :
    else :
        # 1st layer group
        model.add(Convolution3D(64, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv1_1',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv1_2',
                                input_shape=input_shape))
        model.add(Convolution3D(64, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv1_3',
                                input_shape=input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv2_1'))
        model.add(Convolution3D(128, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv2_2'))
        model.add(Convolution3D(128, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv2_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv3a_1'))
        model.add(Convolution3D(256, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv3a_2'))
        model.add(Convolution3D(256, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv3a_3'))
        model.add(Convolution3D(256, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv3b_1'))
        model.add(Convolution3D(256, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv3b_2'))
        model.add(Convolution3D(256, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv3b_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv4a_1'))
        model.add(Convolution3D(512, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv4a_2'))
        model.add(Convolution3D(512, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv4a_3'))
        model.add(Convolution3D(512, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv4b_1'))
        model.add(Convolution3D(512, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv4b_2'))
        model.add(Convolution3D(512, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv4b_3'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv5a_1'))
        model.add(Convolution3D(512, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv5a_2'))
        model.add(Convolution3D(512, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv5a_3'))
        model.add(Convolution3D(512, 1, 1, 3, activation='relu',
                                border_mode='same', name='conv5b_1'))
        model.add(Convolution3D(512, 1, 3, 1, activation='relu',
                                border_mode='same', name='conv5b_2'))
        model.add(Convolution3D(512, 3, 1, 1, activation='relu',
                                border_mode='same', name='conv5b_3'))
        model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))

    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(nb_classes, activation='softmax', name='fcfinal'))

    if summary:
        print(model.summary())

    return model

