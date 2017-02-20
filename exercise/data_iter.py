from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import cPickle as pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adagrad
from keras.regularizers import l2, activity_l2



feat_train = open('/Users/rwa56/Downloads/Homework2_data/train_feat.pickle', 'rb')
train_feat = pickle.load(feat_train)
feat_train.close()

label_train = open('/Users/rwa56/Downloads/Homework2_data/train_lab.pickle', 'rb')
train_label = pickle.load(label_train)
label_train.close()

index_file = open('/Users/rwa56/datasets/train_index.txt', 'w')


for idx, val in train_feat:
    img = array_to_img(val)
    img.save('/Users/rwa56/datasets/img_%d.jpg',idx)

for idx, val in train_label:
    index_file.write(val)


index_file.close()
