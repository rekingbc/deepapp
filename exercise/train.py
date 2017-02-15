from __future__ import print_function
import os
import cPickle as pickle


class train:
    def __init__(self):
        feat_train = open('~/Downloads/Homework2_data/train_feat.pickle', 'rb')
        train_feat = pickle.load(feat_train)
        label_train = open('~/Downloads/Homework2_data/train_lab.pickle', 'rb')
        train_label = pickle.load(feat_train)

        feat_valid = open('~/Downloads/Homework2_data/validation_feat.pickle', 'rb')
        valid_feat = pickle.load(feat_valid)

        label_valid = open('~/Downloads/Homework2_data/validation_lab.pickle', 'rb')
        train_label = pickle.load(feat_valid)



class test:
    def __init__(self):
        feat_test = open('~/Downloads/Homework2_data/train_feat.pickle', 'rb')
        test_feat = pickle.load(feat_test)

        label_test = open('~/Downloads/Homework2_data/test_lab.pickle', 'rb')
        test_label = pickle.load(label_test)
