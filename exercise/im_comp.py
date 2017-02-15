from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import cPickle as pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


batch_size = 12
nb_classes = 10
nb_epoch = 200
data_augmentation = True
img_rows, img_cols = 32, 32
img_channels = 3




feat_train = open('/Users/rwa56//Downloads/Homework2_data/train_feat.pickle', 'rb')
train_feat = pickle.load(feat_train)
feat_train.close()

label_train = open('/Users/rwa56//Downloads/Homework2_data/train_lab.pickle', 'rb')
train_label = pickle.load(label_train)
label_train.close()


feat_valid = open('/Users/rwa56//Downloads/Homework2_data/validation_feat.pickle', 'rb')
valid_feat = pickle.load(feat_valid)
feat_valid.close()

label_valid = open('/Users/rwa56//Downloads/Homework2_data/validation_lab.pickle', 'rb')
valid_label = pickle.load(label_valid)
label_valid.close()


feat_test = open('/Users/rwa56//Downloads/Homework2_data/train_feat.pickle', 'rb')
test_feat = pickle.load(feat_test)
feat_test.close()


print('The train shape: ', train_feat.shape)
print(train_feat.shape[0], 'train samples')

Y_train = np_utils.to_categorical(train_label, nb_classes)
Y_valid = np_utils.to_categorical(valid_label, nb_classes)



model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                                                input_shape=train_feat.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = train_feat.astype('float32')
X_valid = train_valid.astype('float32')
X_train /= 255
X_valid /= 255


if not data_augmentation:
    print('Not using data augmentation.')

else:
    print('Use real-time data augmentation')

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        feature_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(X_train)





label_test = open('~/Downloads/Homework2_data/test_lab.pickle', 'rb')
pickle.dump(test_label, label_test)
label_test.close()





