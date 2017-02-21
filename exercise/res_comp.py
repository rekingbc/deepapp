from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import cPickle as pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import ZeroPadding2D, AveragePooling2D, Convolution2D, MaxPooling2D, merge, Input
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adagrad
from keras.regularizers import l2, activity_l2
import scipy.ndimage.interpolation

from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file







def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             input_tensor=None):
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 32, 32)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (32, 32, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(32, 3, 3, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [32, 32, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 64], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 64], stage=2, block='c')
    x = identity_block(x, 3, [32, 32, 64], stage=2, block='d')

    x = conv_block(x, 3, [64, 64, 128], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 128], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 128], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 128], stage=3, block='d')


    x = AveragePooling2D((3, 3), name='avg_pool')(x)


    if include_top:
        x = Flatten()(x)
        x = Dense(10, activation='softmax', name='cifar10')(x)

    model = Model(img_input, x)

    return model






batch_size = 80
nb_classes = 10
nb_epoch = 50
data_augmentation = True
img_rows, img_cols = 32, 32
img_channels = 3


feat_train = open('/Users/rwa56/Downloads/Homework2_data/train_feat.pickle', 'rb')
train_feat = pickle.load(feat_train)
feat_train.close()

label_train = open('/Users/rwa56/Downloads/Homework2_data/train_lab.pickle', 'rb')
train_label = pickle.load(label_train)
label_train.close()

feat_valid = open('/Users/rwa56/Downloads/Homework2_data/validation_feat.pickle', 'rb')
valid_feat = pickle.load(feat_valid)
feat_valid.close()

label_valid = open('/Users/rwa56/Downloads/Homework2_data/validation_lab.pickle', 'rb')
valid_label = pickle.load(label_valid)
label_valid.close()

feat_test = open('/Users/rwa56/Downloads/Homework2_data/test_feat.pickle', 'rb')
test_feat = pickle.load(feat_test)
feat_test.close()


print('The train shape: ', train_feat.shape)
print(train_feat.shape[0], 'train samples')

print('The validation shape: ', valid_feat.shape)
print(valid_feat.shape[0],'test samples')


print('The test shape: ', test_feat.shape)
print(test_feat.shape[0],'test samples')

X_train = train_feat.astype('float32')
X_valid = valid_feat.astype('float32')
X_test = test_feat.astype('float32')

#X_train /= 255
#X_valid /= 255
#X_test /= 255

Y_train = np_utils.to_categorical(train_label, nb_classes)
Y_valid = np_utils.to_categorical(valid_label, nb_classes)


model = ResNet50(include_top=True)

adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])



if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train, batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_split=0.1)

else:
    print('Use real-time data augmentation')

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=True,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_valid,Y_valid))


test_label = model.predict(X_test)

label_test = open('~/Downloads/Homework2_data/test_lab.pickle', 'rb')
pickle.dump(test_label, label_test)
label_test.close()
