from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import cPickle as pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adagrad
from keras.regularizers import l2, activity_l2
import scipy.ndimage.interpolation
import numpy

batch_size = 40
nb_classes = 10
nb_epoch = 3
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

for i in xrange(X_train.shape[0]):
    X_train[i,:,:,:] = scipy.ndimage.interpolation.rotate(X_train[i,:,:,:],-90)

#X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)

for i in xrange(X_valid.shape[0]):
    X_valid[i,:,:,:] = scipy.ndimage.interpolation.rotate(X_valid[i,:,:,:],-90)

#X_valid = X_valid.reshape(X_valid.shape[0], 3, 32, 32)

for i in xrange(X_test.shape[0]):
    X_test[i,:,:,:] = scipy.ndimage.interpolation.rotate(X_test[i,:,:,:],-90)

#X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)


print('Train Sample Shape: ', X_train.shape[1:])

Y_train = np_utils.to_categorical(train_label, nb_classes)
Y_valid = np_utils.to_categorical(valid_label, nb_classes)


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


#model = ResNet50(include_top=True)


adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])



if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train, batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_data=(X_valid,Y_valid))

else:
    print('Use real-time data augmentation')

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=True)

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_valid,Y_valid))


test_label = model.predict(X_test)

label_test = open('/Users/rwa56/Downloads/Homework2_data/test_lab.pickle', 'wb')


for i in range(len(test_label)):
        test_lab[i] = test_label[i]
#save the pickle file that you should upload:


pickle.dump(test_lab, label_test)
label_test.close()
