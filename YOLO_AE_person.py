#!/usr/bin/env python
# coding: utf-8

TEST=0

UNSUP_NUM_IMAGES = 5000 if not TEST else 5
EVAL_NUM_IMAGES  = 100 if not TEST else 5
EPOCHS = 180 if not TEST else 1
SUP=0.2 #SUP is between 0 to 1 represents the relative part of all train samples that are supervised

# todo:
# todo: 1. add the pre-trained weights, randomize first and last. use a new label (more images)
'constant randomness:'
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# this code is a modification of:
# notes:
# todo : I canceled the randomize weights for the last layer + freezed the weights for all of the layers (some weights were trained anyway, maybe i need to also untrain the norm layers).

# **Outline of Steps**
#     + Initialization
#         + Download COCO detection data from http://cocodataset.org/#download
#             + http://images.cocodataset.org/zips/train2014.zip <= train images
#             + http://images.cocodataset.org/zips/val2014.zip <= validation images
#             + http://images.cocodataset.org/annotations/annotations_trainval2014.zip <= train and validation annotations
#         + Run this script to convert annotations in COCO format to VOC format
#             + https://gist.github.com/chicham/6ed3842d0d2014987186#file-coco2pascal-py
#         + Download pre-trained weights from https://pjreddie.com/darknet/yolo/
#             + https://pjreddie.com/media/files/yolo.weights
#         + Specify the directory of train annotations (train_annot_folder) and train images (train_image_folder)
#         + Specify the directory of validation annotations (valid_annot_folder) and validation images (valid_image_folder)
#         + Specity the path of pre-trained weights by setting variable *wt_path*
#     + Construct equivalent network in Keras
#         + Network arch from https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg
#     + Load the pretrained weights
#     + Perform training 
#     + Perform detection on an image with newly trained weights
#     + Perform detection on an video with newly trained weights

# # Initialization

# In[51]:
#from IPython import get_ipython
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, \
    UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']

IMAGE_H, IMAGE_W = 416, 416
GRID_H, GRID_W = 13, 13
BOX = 5
CLASS = len(LABELS)
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD = 0.3  # 0.5  # confidence value consider as positive
NMS_THRESHOLD = 0.3  # 0.45 # IOU value threshold
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0

BATCH_SIZE = 16 if not TEST else 1 #16 originally
WARM_UP_BATCHES = 0
TRUE_BOX_BUFFER = 50

MAX_BOX_PER_IMAGE = 10


# In[53]:


wt_path = 'yolov2-voc.weights'
train_image_folder = './data/images/train2014/'
train_annot_folder = './data/train_converted/'
valid_image_folder = './data/images/val2014/'
valid_annot_folder = './data/val_converted/'



# # Construct the network

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


import frontend
""" creates a new dir names coco_x with the results, weights, and all the relevant files"""
TB_COUNT = len([d for d in os.listdir(os.path.expanduser('./results_person/')) if 'cococo_' in d]) + 1
PATH = os.path.expanduser('./results_person/') + 'cococo_' + '_' + str(TB_COUNT)
os.makedirs(PATH)
# PATH = "./results_person/results_person/coco__2"
print("=================== Directory " , PATH ,  " Created ")

class ToharGenerator(BatchGenerator):
    def __getitem__(self, item):
        t = super().__getitem__(item)[0]
        return t[0], t[0]


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

# ############TINY YOLO
# # Layer 1
# x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
# x = BatchNormalization(name='norm_1')(x)
# x = LeakyReLU(alpha=0.1)(x)
# encoded = MaxPooling2D(pool_size=(2, 2))(x)
#
#
# # autoencoder:
#
# y = Conv2D(16, (3, 3), strides=(1, 1), padding='same',name='decoder_conv_1')(encoded)
# y = UpSampling2D((2, 2))(y)
# decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same',name='decoder_conv_2')(y)
#
# autoencoder = Model(input_image, decoded)
# autoencoder.summary()
# autoencoder.compile(optimizer='adam', loss='mse')
# # end autoencoder
#
# # Layer 2
# i = 0
# x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 2), use_bias=False)(encoded)
# x = BatchNormalization(name='norm_' + str(i + 2))(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
#
# # Layer 3 - 5
# for i in range(1, 4):
#     x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 2), use_bias=False)(x)
#     x = BatchNormalization(name='norm_' + str(i + 2))(x)
#     x = LeakyReLU(alpha=0.1)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#
# # Layer 6
# x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
# x = BatchNormalization(name='norm_6')(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
#
# # Layer 7 - 8
# for i in range(0, 2):
#     x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 7), use_bias=False)(x)
#     x = BatchNormalization(name='norm_' + str(i + 7))(x)
#     x = LeakyReLU(alpha=0.1)(x)
# ########### END TINY YOLO

######## FULL YOLO ##########
# Layer 1
x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
encoded = MaxPooling2D(pool_size=(2, 2))(x)

# autoencoder:

y = Conv2D(32, (3, 3), strides=(1, 1), padding='same',name='decoder_conv_1')(encoded)
y = UpSampling2D((2, 2))(y)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same',name='decoder_conv_2')(y)

autoencoder = Model(input_image, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mse')
# end autoencoder

# FULL YOLO
# Layer 2
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(encoded)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)


# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

#
#
# input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
# true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))
#
# # Layer 1
# x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
# x = BatchNormalization(name='norm_1')(x)
# x = LeakyReLU(alpha=0.1)(x)
# encoded = MaxPooling2D(pool_size=(2, 2))(x)
#
# # autoencoder:
#
# y = Conv2D(32, (3, 3), strides=(1, 1), padding='same',name="decoder")(encoded)
# y = UpSampling2D((2, 2))(y)
# decoded = Conv2D(3, (3, 3), activation='sigmoid',name="decoder2", padding='same')(y)
#
# autoencoder = Model(input_image, decoded)
# autoencoder.summary()
# autoencoder.compile(optimizer='adam', loss='mse')
# # end autoencoder
#
#
# # Layer 2
# x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False, trainable=False)(encoded)
# x = BatchNormalization(name='norm_2', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
#
# # Layer 3
# x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_3', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 4
# x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_4', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 5
# x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_5', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
#
# # Layer 6
# x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_6', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 7
# x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_7', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 8
# x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_8', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
#
# # Layer 9
# x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_9', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 10
# x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_10', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 11
# x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_11', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 12
# x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_12', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 13
# x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_13', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# skip_connection = x
#
# x = MaxPooling2D(pool_size=(2, 2))(x)
#
# # Layer 14
# x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_14', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 15
# x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_15', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 16
# x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_16', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 17
# x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_17', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 18
# x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_18', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 19
# x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_19', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 20
# x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_20', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 21
# skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False, trainable=False)(
#     skip_connection)
# skip_connection = BatchNormalization(name='norm_21', trainable=False)(skip_connection)
# skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
# skip_connection = Lambda(space_to_depth_x2)(skip_connection)
#
# # x = skip_connection
# x = concatenate([skip_connection, x])
#
# # Layer 22
# x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False, trainable=False)(x)
# x = BatchNormalization(name='norm_22', trainable=False)(x)
# x = LeakyReLU(alpha=0.1)(x)
#
# # Layer 23
# x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
# output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
# output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

model.summary()

yolo = Model([input_image, true_boxes], output)


'''Load pre-trained weights'''
# Load pretrained weights

# **Load the weights originally provided by YOLO**
print("**Load the weights originally provided by YOLO**")
weight_reader = WeightReader(wt_path)

weight_reader.reset()  # don't worry! it doesn't delete the weights.
nb_conv = 23

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv_' + str(i))

    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])




# **Randomize weights of the last layer**
# print("=====")
#
# # for layer in model.layers: print(layer.get_config(), layer.get_weights())
# print(model.layers[0])
# for layer in model.layers: print(layer.get_config())
# print("=====")
# layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
#
# output_layer = layer_dict[layer_name]
# print("========randomize last layer")
# model.layers[0]
# # TO get first four layers
# model.layers[0:3]
# #To get the input shape
# model.layers[layer_of_interest_index].input_shape
# #To get the input shape
# model.layers[layer_of_interest_index].output_shape
# TO get weights matrices
# model.layers[layer_of_interest_index].get_weights()
# print("======== original last layer is: ========")


'''randomize first layer weight'''

firstlayer = model.get_layer('conv_1')
# print(layer.get_config())

weights = firstlayer.get_weights()
# before = firstlayer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
# new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

firstlayer.set_weights([new_kernel])
# after = firstlayer.get_weights()

# norm_1 = model.get_layer('norm_1')
# weights = norm_1.get_weights()
#
# # size = np.prod(norm_layer.get_weights()[0].shape)
# #
# # beta = weight_reader.read_bytes(size)
# # gamma = weight_reader.read_bytes(size)
# # mean = weight_reader.read_bytes(size)
# # var = weight_reader.read_bytes(size)
# #
# # weights = norm_layer.set_weights([gamma, beta, mean, var])
#
# new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
# norm_1.set_weights([new_kernel])
#
#
# "radomize last layer"
# layer   = model.layers[-4] # the last convolutional layer
# #equivalent to:
# #later = model.get_layer('conv_23') #to get the config: later = model.get_layer('conv_23').get_config()
# # print(layer.get_config())
# weights = layer.get_weights()
# # print(weights, layer)
# new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
# new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)
#
# layer.set_weights([new_kernel, new_bias])


'''Perform training'''


import backend

def path(path):
    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(path):
        os.makedirs(path)
        # print("Directory ", path, " Created ")
    else:
        pass
        # print("Directory ", path, " already exists")
    return path

def predict(model, image, i, img_name, path=""):
    """
    input_size = IMAGE_H

    image_h, image_w, _ = image.shape
    feature_extractor = backend.FullYoloFeature()
    image = cv2.resize(image, (input_size, input_size))
    image =feature_extractor.normalize(image)

    input_image = image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    dummy_array = np.zeros((1,1,1,1, MAX_BOX_PER_IMAGE,4))

    netout = model.predict([input_image, dummy_array])[0]
    boxes  = decode_netout(netout, ANCHORS, len(LABELS))
    """
    dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))
    # print("dummy array:", dummy_array)
    plt.figure(figsize=(10, 10))

    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)

    netout = model.predict([input_image, dummy_array])

    boxes = decode_netout(netout[0],
                          obj_threshold=OBJ_THRESHOLD,
                          nms_threshold=NMS_THRESHOLD,
                          anchors=ANCHORS,
                          nb_class=CLASS)
    image = draw_boxes(image, boxes, labels=LABELS)

    plt.imshow(image[:, :, ::-1])
    path = str(path)
    if i <= 100:
        # Create target directory & all intermediate directories if don't exists
        if not os.path.exists(path):
            os.makedirs(path)
            # print("Directory ", path, " Created ")
        else:
            pass
            # print("Directory ", path, " already exists")
        plt.savefig(path+ "/" + img_name)

    return boxes

from utils import decode_netout, compute_overlap, compute_ap

from os.path import normpath, basename

def evaluate(model, generator,
             iou_threshold=0.3,
             score_threshold=0.3,
             max_detections=100,
             save_path=None):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)

        path = generator.images[i]['filename']
        img_name = basename(normpath(path))

        raw_height, raw_width, raw_channels = raw_image.shape

        # make the boxes and the labels
        pred_boxes = predict(model, raw_image, i, img_name, path=save_path)

        score = np.array([box.score for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                    box.ymax * raw_height, box.score] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

            # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes = pred_boxes[score_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

    # compute mAP by comparing all detections and all annotations
    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision


    # import pickle
    # f = open(save_path+"/mAP.pkl", "wb")
    # pickle.dump(average_precisions, f)
    # f.close()

    return average_precisions


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    """
    Adjust prediction
    """
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

    ### adjust w and h
    true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES),
                                                   lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                            true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                                                                ANCHORS, [1, 1, 1, BOX, 2]) * no_boxes_mask,
                                                            tf.ones_like(coord_mask)],
                                                   lambda: [true_box_xy,
                                                            true_box_wh,
                                                            coord_mask])

    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """
    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall)

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

    return loss


'''DATASET'''
# **Parse the annotations to construct train generator and validation generator**

generator_config = {
    'IMAGE_H': IMAGE_H,
    'IMAGE_W': IMAGE_W,
    'GRID_H': GRID_H,
    'GRID_W': GRID_W,
    'BOX': BOX,
    'LABELS': LABELS,
    'CLASS': len(LABELS),
    'ANCHORS': ANCHORS,
    'BATCH_SIZE': BATCH_SIZE,
    'TRUE_BOX_BUFFER': 50,
}

def normalize(image):
    return image / 255.
train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
## write parsed annotations to pickle for fast retrieval next time
with open('train_imgs_person', 'wb') as fp:
   pickle.dump(train_imgs, fp)


## read saved pickle of parsed annotations
with open('train_imgs_person', 'rb') as fp:
    train_imgs = pickle.load(fp)

from random import shuffle
shuffle(train_imgs)

with open('train_imgs_shuffled_person', 'wb') as fp:
   pickle.dump(train_imgs, fp)

with open('train_imgs_shuffled_person', 'rb') as fp:
    train_imgs = pickle.load(fp)

valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=LABELS)
## write parsed annotations to pickle for fast retrieval next time
with open('valid_imgs_person', 'wb') as fp:
   pickle.dump(valid_imgs, fp)
# read saved pickle of parsed annotations

with open('valid_imgs_person', 'rb') as fp:
    valid_imgs = pickle.load(fp)



#train_imgs is our training set (supervised+unsupervised) for the AE

train_imgs = train_imgs[:UNSUP_NUM_IMAGES]
train_valid_split = int(0.8*len(train_imgs))
train = train_imgs[:train_valid_split] #AE
valid = train_imgs[train_valid_split:] #AE

#take SUP persentage from train dataset to be the supervised dataset
sup = train[:int(SUP*len(train))]

s_train_val_split = int(0.8*len(sup))

train_sup = sup[:s_train_val_split] #model
val_sup   = sup[s_train_val_split:] #model

# splits
# sup_train_imgs = train_imgs[:SUP_NUM_IMAGES]
# # split the training set (supervised date) into train and validation 80%, 20% respectively:
# train = sup_train_imgs[:int(SUP_NUM_IMAGES*0.8)]
# val = sup_train_imgs[-int(SUP_NUM_IMAGES*0.2):] #takes the last 20% images from the training
# ae_unsup = train_imgs[-UNSUP_NUM_IMAGES:]
# ae_train = ae_unsup[:int(UNSUP_NUM_IMAGES*0.8)]
# ae_val = ae_unsup[-int(UNSUP_NUM_IMAGES*0.2):]

train_batch = BatchGenerator(train_sup, generator_config, norm=normalize)

valid_batch = BatchGenerator(val_sup, generator_config, norm=normalize, jitter=False)

#for the AE:
"""we use the unsupervised data to train the AE (which we get from the end of the training set"""
#todo: play with the jitter -- input true output false
tohar_train_batch = ToharGenerator(train, generator_config, norm=normalize,
                                   jitter=False)  # outputs (input,input) rather than (input, ground truth)
tohar_valid_batch = ToharGenerator(valid, generator_config, norm=normalize,
                                   jitter=False)  # outputs (input,input) rather than (input, ground truth)

"""we evaluate the model on the original validation set"""
eval_imgs = valid_imgs[:EVAL_NUM_IMAGES] #we use the valid_imgs as our evaluation set (testing). while we use 20% of the training for validation.
tohar_eval_batch = BatchGenerator(eval_imgs, generator_config, norm=normalize, jitter=False,
                                  shuffle=False)  # outputs (input,input) rather than (input, ground truth)

# **Setup a few callbacks and start the training**

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=30,
                           mode='min',
                           verbose=1)

ae_checkpoint = ModelCheckpoint(PATH+'/AE_weights_coco.h5',
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min',
                                period=1)

checkpoint = ModelCheckpoint(PATH+'/weights_coco.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)

t_checkpoint = ModelCheckpoint(PATH+'/T_weights_coco.h5',
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min',
                                period=1)

path('./logs/AE/')
path('./logs/WAE/')
path('./logs/T/')
ae_tb_counter = len([log for log in os.listdir(os.path.expanduser('./logs/AE/')) if 'cococo_' in log]) + 1
ae_tensorboard = TensorBoard(log_dir=os.path.expanduser('./logs/AE/') + 'cococo_' + '_' + str(ae_tb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

tb_counter = len([log for log in os.listdir(os.path.expanduser('./logs/WAE')) if 'cococo_' in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('./logs/WAE/') + 'cococo_' + '_' + str(tb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

ttb_counter = len([log for log in os.listdir(os.path.expanduser('./logs/T/')) if 'cococo_' in log]) + 1
ttensorboard = TensorBoard(log_dir=os.path.expanduser('./logs/T/') + 'cococo_' + '_' + str(ttb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
# optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)
yolo.compile(loss=custom_loss, optimizer=optimizer)

##AE:
from keras.callbacks import TensorBoard

history = autoencoder.fit_generator(generator=tohar_train_batch,  # train_batch #(input, input)
                          steps_per_epoch=len(tohar_train_batch),
                          epochs=EPOCHS,
                          verbose=1,
                          validation_data=tohar_valid_batch,
                          validation_steps=len(tohar_valid_batch),
                          callbacks=[early_stop, ae_checkpoint, ae_tensorboard],
                          max_queue_size=3)
print(history.history.keys())
# history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(path(PATH+"/AE")+"/AE_plot.jpg")
loss = history.history['loss']
val_loss = history.history['val_loss']
l = np.array(loss)
v = np.array(val_loss)

f = open(PATH + "/logs.txt", "w")
f.write("AE:")
f.write(str({'l': l,'v': v}))
f.write('\n')


print("===================== Done training AE")


print("===================== Load weights from AE_weights_coco.h5 to model")

model.load_weights(PATH+'/AE_weights_coco.h5',
                   by_name=True)  # copy the AE's weights to the "YOLO model" weights, only to layers with the same name as the AE

## end ae

print("===================== Start fitting YOLO model")
history2 = model.fit_generator(generator=train_batch,  # train_batch #(input, ground_truth)
                    steps_per_epoch=len(train_batch),
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=valid_batch,
                    validation_steps=len(valid_batch),
                    callbacks=[early_stop, checkpoint, tensorboard],
                    max_queue_size=3)


plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(path(PATH+"/W_AE")+"/wAE_plot.jpg")

loss = history2.history['loss']
val_loss = history2.history['val_loss']
l = np.array(loss)
v = np.array(val_loss)


f.write("MODEL w AE:")
f.write(str({'l': l,'v': v}))
f.write('\n')

# Perform detection on image

# print("===================== load YOLO model's weights to weights_coco.h5")

# evaluate:

#take the best weights (not neccecerely the current weights of the model)
model.load_weights(PATH+"/weights_coco.h5")
"""evaluating on AE+YOLO (original weights from YOLO, then pre-trining with AE (big unsupervised set), then training with small supervised dataset)"""
AE = evaluate(model, tohar_eval_batch, save_path=PATH+"/W_AE")
print("AE:\n",AE)
print(np.average(list(AE.values())))


#
"""evaluating on only YOLO with training on small supervised datatset"""

print("===================== Start fitting model_t -- trained without AE")
yhistory = yolo.fit_generator(generator=train_batch,  # train_batch #(input, ground_truth)
                    steps_per_epoch=len(train_batch),
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=valid_batch,
                    validation_steps=len(valid_batch),
                    callbacks=[early_stop, t_checkpoint, ttensorboard],
                    max_queue_size=3)


plt.plot(yhistory.history['loss'])
plt.plot(yhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(path(PATH+"/NO_AE")+"/NO_AE_plot.jpg")

yolo.load_weights(PATH+"/T_weights_coco.h5")
NO_AE = evaluate(yolo, tohar_eval_batch, save_path=PATH+"/NO_AE")
print("NO_AE:\n", NO_AE)
print(np.average(list(NO_AE.values())))

loss = yhistory.history['loss']
val_loss = yhistory.history['val_loss']
l = np.array(loss)
v = np.array(val_loss)

f.write("NO AE:")
f.write(str({'l': l,'v': v}))
f.write('\n')
f.close()

#
# # """evaluating on original YOLO (no training at all)"""
# # # model.load_weights("yolo.h5")
# # YOLO = evaluate(model_un, tohar_eval_batch, save_path=PATH+"/YOLO") #model_un is same as model but untrained at all
# # print("YOLO:\n", YOLO)
# # print(np.average(list(YOLO.values())))
#
#
print("model w AE:")
print(np.average(list(AE.values())))
print("NO_AE:")
print(np.average(list(NO_AE.values())))
# print("YOLO:")
# print(np.average(list(YOLO.values())))


params={'sup to unsup ratio:':SUP,
        'all train:':len(train),
        'supervised:': len(sup),
        "UNSUP_NUM_IMAGES(=all train):":UNSUP_NUM_IMAGES,
        "EVAL_NUM_IMAGES:":EVAL_NUM_IMAGES,
        "EPOCHS": EPOCHS}

f = open(PATH + "/mAP.txt", "w")
f.write("AE:")
f.write(str(AE)+"\n")
f.write("NO_AE:")
f.write(str(NO_AE)+"\n")
# f.write("YOLO:")
# # f.write(str(YOLO)+"\n")
f.write("AVG:")
f.write(str(np.average(list(AE.values())))+"\n")
f.write(str(np.average(list(NO_AE.values())))+"\n")
# # f.write(str(np.average(list(YOLO.values())))+"\n")
f.write("LOG:"+"\n")
f.write(str(params) )
f.write("all coco detection,  init with pre trained weights from VOC , randomize first layer, not randomize last layer")
#
f.close()

exit()
model.load_weights("./results_person/results_person/coco__2/weights_coco.h5")
image = cv2.imread('./data/images/val2014/COCO_val2014_000000000139.jpg')
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
plt.figure(figsize=(10,10))

input_image = cv2.resize(image, (416, 416))
input_image = input_image / 255.
input_image = input_image[:,:,::-1]
input_image = np.expand_dims(input_image, 0)

netout = model.predict([input_image, dummy_array])

boxes = decode_netout(netout[0],
                      obj_threshold=0.3001,
                      nms_threshold=0.3,
                      anchors=ANCHORS,
                      nb_class=CLASS)

image1 = draw_boxes(image, boxes, labels=LABELS)

plt.imshow(image1[:,:,::-1]); #plt.show()

plt.savefig("./results_person/results_person/coco__2/figure_AE")

netout = yolo.predict([input_image, dummy_array])

boxes = decode_netout(netout[0],
                      obj_threshold=0.309,
                      nms_threshold=0.3,
                      anchors=ANCHORS,
                      nb_class=CLASS)

image2 = draw_boxes(image, boxes, labels=LABELS)

plt.imshow(image2[:,:,::-1]); #plt.show()

plt.savefig("./results_person/results_person/coco__2/figure_T")
print("\a")













print("stop here before videos")
exit()

# # Perform detection on video

# In[ ]:


# model.load_weights("weights_coco.h5")

dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))

# In[ ]:


video_inp = '../basic-yolo-keras/images/phnom_penh.mp4'
video_out = '../basic-yolo-keras/images/phnom_penh_bbox.mp4'

video_reader = cv2.VideoCapture(video_inp)

nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'XVID'),
                               50.0,
                               (frame_w, frame_h))

for i in tqdm(range(nb_frames)):
    ret, image = video_reader.read()

    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)

    netout = model.predict([input_image, dummy_array])

    boxes = decode_netout(netout[0],
                          obj_threshold=0.3,
                          nms_threshold=NMS_THRESHOLD,
                          anchors=ANCHORS,
                          nb_class=CLASS)
    image = draw_boxes(image, boxes, labels=LABELS)

    video_writer.write(np.uint8(image))

video_reader.release()
video_writer.release()
