import os

from scipy import ndimage

import numpy as np
import pandas as pd

import h5py as h5py
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.models import Sequential, model_from_json


def input_images(test_path='../data/images/', img_width=270, img_height=270):
    labels = pd.read_csv('../data/labels_for_class0_and_class1.csv', header=0)
    testxs = []
    testys = []

    print("Reading images and assigning labels ...")
    for file in os.listdir(test_path):
        file_name = file.replace('.jpeg', '')
        if file[:1] != '.':
            im = ndimage.imread(test_path + file)
            lab = labels[labels['image'] == file_name]['level']
            testxs.append(im)
            testys.append(lab)
    print(len(testxs))
    print(len(testys))
    X_test = np.reshape(testxs, [200, 3, 270, 270])
    # mean_image = np.mean(X_test, axis=0)
    # X_test = X_test - mean_image.astype(np.uint8)
    Y_test = np.concatenate(testys)

    return X_test, Y_test


def load_vgg16_model(weights_path='../data/models/vgg16_weights.h5', img_height=270,
                     img_width=270):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    return model


def load_top_model(top_model_weights_path='../data/models/top_model_weights.h5',
                   top_model_json_path='../data/models/model.json'):
    json_file = open(top_model_json_path, 'r')
    top_model_json = json_file.read()
    json_file.close()
    top_model = model_from_json(top_model_json)
    top_model.load_weights(top_model_weights_path)
    return top_model
