#!/usr/bin/env python
# coding: utf-8

# this code is a modification of:
# notes:
# todo : I canceled the randomize weights for the last layer + freezed the weights for all of the layers (some weights were trained anyway).
#todo : mayb -- save to fule during evaluate function the outputs
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
    UpSampling2D, TimeDistributed, LSTM
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
from preprocessing import parse_annotation, BatchGenerator, LSTMBatchGenerator
from utils import WeightReader, decode_netout, draw_boxes

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:
SUP_NUM_IMAGES   = 3
UNSUP_NUM_IMAGES = 3
EVAL_NUM_IMAGES  = 3


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
OBJ_THRESHOLD = 0.3  # 0.5
NMS_THRESHOLD = 0.3  # 0.45
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0

BATCH_SIZE = 16
WARM_UP_BATCHES = 0
TRUE_BOX_BUFFER = 50

MAX_BOX_PER_IMAGE = 10

# In[53]:


wt_path = 'yolov2.weights'
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
# TB_COUNT = len([d for d in os.listdir(os.path.expanduser('./results_lstm/')) if 'coco_' in d]) + 1
# PATH = os.path.expanduser('./results_lstm/') + 'coco_' + '_' + str(TB_COUNT)
# os.makedirs(PATH)
PATH = './lstm/'
print("=================== Directory " , PATH ,  " Created ")
# PATH = "./results/coco__25"



class ToharGenerator2(BatchGenerator):
    def __getitem__(self, item):
        # t= [x_batch,b_batch],y_batch
        #    [input,goutndtruth],desired network output]
        t = super().__getitem__(item)
        x_batch = t[0][0] #the input
        GT = t[0][1]
        y_batch = t[1]

        new_x_batch = predict(model,x_batch) #instead of input img vector we want the YOLO's output vector
        t[0][0]= new_x_batch
        return [new_x_batch, GT], y_batch



input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

# Layer 1
x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
encoded = MaxPooling2D(pool_size=(2, 2))(x)


# Layer 2
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False, trainable=False)(encoded)
x = BatchNormalization(name='norm_2', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_3', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_4', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_5', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_6', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_7', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_8', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_9', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_10', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_11', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_12', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_13', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_14', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_15', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_16', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_17', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_18', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_19', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_20', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False, trainable=False)(
    skip_connection)
skip_connection = BatchNormalization(name='norm_21', trainable=False)(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False, trainable=False)(x)
x = BatchNormalization(name='norm_22', trainable=False)(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 23
x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

# small hack to allow true_boxes to be registered when Keras build the model
# for more information: https://github.com/fchollet/keras/issues/2790
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

# model.summary()
print("output=====")
print(output.shape)
'''build lstm model: '''
lstm_input = Input(shape=(GRID_H, GRID_W, BOX, 4 + 1 + CLASS))

input_dim = GRID_H * GRID_W * BOX * (4 + 1 + CLASS)
# input_dim=(GRID_H,GRID_W, BOX, 4 + 1 + CLASS, 1, 1, 1, TRUE_BOX_BUFFER, 4)
print(input_dim)

timesteps = EVAL_NUM_IMAGES


# lstm.add(units= Dense(input_shape=(GRID_H, GRID_W, BOX, 4 + 1 + CLASS)))
# l=Lambda(lambda x: K.batch_flatten(x))(lstm_input)
# l=LSTM(input_dim, batch_input_shape= (None, timesteps, input_dim), activation='sigmoid',recurrent_activation='hard_sigmoid',return_sequences=True)(l)
# # l = (Dense(output_dim=input_dim, activation="relu"))(lstm)
# #
# # # l = LSTM(input_dim)(l)
# # # # hidden_layer = Dense(output_dim=input_shape, activation="relu")(x)
# # # # outputs = Dense(output_dim=input_shape, activation="softmax")(hidden_layer)
# #
# loutput = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(l)
# #
# # # small hack to allow true_boxes to be registered when Keras build the model
# # # for more information: https://github.com/fchollet/keras/issues/2790
# out = Lambda(lambda args: args[0])([loutput, true_boxes])
#
#
#
# lstm = Model([lstm_input, true_boxes], out)
# lstm.summary()

input_dim = GRID_H * GRID_W * BOX * (4 + 1 + CLASS)

#take 5 frames every time
frames = Input(shape=(5, IMAGE_H, IMAGE_W, 3))
x = TimeDistributed(model)(frames)
x = TimeDistributed(Flatten())(x)
#now- timestamsp=5
x = LSTM(input_dim, name='lstm')(x)
out = Dense(input_dim, name='out')(x)
lstm = Model(inputs=frames, outputs=out)

exit()
# # Load pretrained weights

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

# model_t = model  #model that trained but not pre-trained
# model_un = model #model without training at all

# **Randomize weights of the last layer**

# In[ ]:

# print("========randomize last layer")
# layer   = model.layers[-4] # the last convolutional layer
# weights = layer.get_weights()
#
# new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
# new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)
#
# layer.set_weights([new_kernel, new_bias])


# # Perform training

# **Loss function**

# $$\begin{multline}
# \lambda_\textbf{coord}
# \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#      L_{ij}^{\text{obj}}
#             \left[
#             \left(
#                 x_i - \hat{x}_i
#             \right)^2 +
#             \left(
#                 y_i - \hat{y}_i
#             \right)^2
#             \right]
# \\
# + \lambda_\textbf{coord}
# \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#          L_{ij}^{\text{obj}}
#          \left[
#         \left(
#             \sqrt{w_i} - \sqrt{\hat{w}_i}
#         \right)^2 +
#         \left(
#             \sqrt{h_i} - \sqrt{\hat{h}_i}
#         \right)^2
#         \right]
# \\
# + \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#         L_{ij}^{\text{obj}}
#         \left(
#             C_i - \hat{C}_i
#         \right)^2
# \\
# + \lambda_\textrm{noobj}
# \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#     L_{ij}^{\text{noobj}}
#         \left(
#             C_i - \hat{C}_i
#         \right)^2
# \\
# + \sum_{i = 0}^{S^2}
# L_i^{\text{obj}}
#     \sum_{c \in \textrm{classes}}
#         \left(
#             p_i(c) - \hat{p}_i(c)
#         \right)^2
# \end{multline}$$

# In[ ]:

import backend


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
            print("Directory ", path, " Created ")
        else:
            pass
            # print("Directory ", path, " already exists")
        #os.makedirs(path) # create the directory on given path, also if any intermediate-level directory don’t exists then it will create that too.
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


    import pickle
    f = open(save_path+"/mAP.pkl", "wb")
    pickle.dump(average_precisions, f)
    f.close()

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



# train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
# ## write parsed annotations to pickle for fast retrieval next time
# with open('train_imgs', 'wb') as fp:
#    pickle.dump(train_imgs, fp)


# ## read saved pickle of parsed annotations
# with open('train_imgs', 'rb') as fp:
#     train_imgs = pickle.load(fp)
#
# from random import shuffle
# shuffle(train_imgs)
#
# with open('train_imgs_shuffled', 'wb') as fp:
#    pickle.dump(train_imgs, fp)

with open('train_imgs_shuffled', 'rb') as fp:
    train_imgs = pickle.load(fp)

# valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=LABELS)
# ## write parsed annotations to pickle for fast retrieval next time
# with open('valid_imgs', 'wb') as fp:
#    pickle.dump(valid_imgs, fp)
## read saved pickle of parsed annotations
with open('valid_imgs', 'rb') as fp:
    valid_imgs = pickle.load(fp)

sup_train_imgs = train_imgs[:SUP_NUM_IMAGES]

# split the training set (supervised date) into train and validation 80%, 20% respectively:
train = sup_train_imgs[:int(SUP_NUM_IMAGES*0.8)]
val = sup_train_imgs[-int(SUP_NUM_IMAGES*0.2):] #takes the last 20% images from the training

train_batch = BatchGenerator(train, generator_config, norm=normalize)

eval_imgs = valid_imgs[:EVAL_NUM_IMAGES] #we use the valid_imgs as our evaluation set (testing). while we use 20% of the training for validation.

valid_batch = BatchGenerator(val, generator_config, norm=normalize, jitter=False)



"""we evaluate the model on the validation set"""
tohar_eval_batch = BatchGenerator(eval_imgs, generator_config, norm=normalize, jitter=False,
                                  shuffle=False)




# **Setup a few callbacks and start the training**




early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint(PATH+'/LSTM_weights_coco.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)
org_checkpoint = ModelCheckpoint(PATH+'/original_weights_coco.h5',
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min',
                                period=1)


# In[ ]:


tb_counter = len([log for log in os.listdir(os.path.expanduser('./lstm/')) if 'coco_' in log]) + 1
tensorboard = TensorBoard(log_dir=os.path.expanduser('./lstm/') + 'coco_' + '_' + str(tb_counter),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
# optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)
# model_t.compile(loss=custom_loss, optimizer=optimizer)

from keras.callbacks import TensorBoard

"""evaluating on original YOLO (no training at all)"""
model.load_weights("yolo.h5")
# YOLO = evaluate(model, tohar_eval_batch, save_path=PATH+"/YOLO")
# print("YOLO:\n", YOLO)
# print(np.average(list(YOLO.values())))

'''creating a modified batch to the lstm:'''
# [x_batch, GT], y_batch
# [x_batch, GT], \
lstm_batch = LSTMBatchGenerator(eval_imgs, generator_config, model, norm=None, jitter=False, shuffle=False)


print(len(lstm_batch))



exit()

"""X_train2 should be YOLO's output vectors
y_train2 should be the ground truth in the exact same format of YOLO's output
"""

# autoencoder.fit_generator(generator=train_batch_lstm,  #(input, input)
#                           steps_per_epoch=len(train_batch_lstm),
#                           epochs=100,
#                           verbose=1,
#                           # validation_data=tohar_valid_batch,
#                           # validation_steps=len(tohar_valid_batch),
#                           callbacks=[early_stop, ae_checkpoint, tensorboard],
#                           max_queue_size=3)
# print("===================== Done training AE")
# print("===================== Save weights to AE_weights_coco.h5")
# autoencoder.save_weights(PATH+"/AE_weights_coco.h5")  # save the autoencoder's weights in this file
# print("===================== Load weights from AE_weights_coco.h5")
# model.load_weights(PATH+'/AE_weights_coco.h5',
#                    by_name=True)  # copy the AE's weights to the "YOLO model" weights, only to layers with the same name as the AE

## end ae
##uncomment for training:

# Perform detection on image

# print("===================== load YOLO model's weights to weights_coco.h5")

# evaluate:

# train_batch_lstm = ToharGenerator2(train, generator_config, norm=normalize)

""" Add lstm on top of the trained YOLO model. the lstm should have many to many sturcture. each latm cell predict 1 output . help:"""
# https://stackoverflow.com/questions/49535488/lstm-on-top-of-a-pre-trained-cnn
# https://github.com/keras-team/keras/issues/5527
''' Freeze previous layers '''
for layer in model.layers:
    layer.trainable = False

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam

frames = len(tohar_eval_batch)
print(frames)

units = GRID_H * GRID_W * BOX * (4 + 1 + CLASS)
print("==========",units)
length=5 #todo:batch size

#todo: input dim is problematic.
# input_images = Input(shape=( None, frames ,IMAGE_H, IMAGE_W, 3))
#https://riptutorial.com/keras/example/29812/vgg-16-cnn-and-lstm-for-video-classification

# frames, rows, columns, channels = 10, IMAGE_H, IMAGE_W, 3
# video = Input(shape=(frames,
#                      rows,
#                      columns,
#                      channels))
#
# # cnn_base = VGG16(input_shape=(rows, columns, channels),
# #                  weights="imagenet",
# #                  include_top=False)
# # cnn_out = GlobalAveragePooling2D()(cnn_base.output)
# # cnn = Model(input=cnn_base.input, output=cnn_out)
#
# model.trainable = False
#
# encoded_frames = TimeDistributed(model)(video)
# encoded_sequence = LSTM(256)(encoded_frames)
# hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
# outputs = Dense(output_dim=units, activation="softmax")(hidden_layer)
# lstm = Model([video], outputs)

#
# # x = Reshape((len(train_batch)*10 ,IMAGE_H, IMAGE_W, 3))(input_images)
# x = TimeDistributed(model)(x)
# x = TimeDistributed(Flatten())(x)
# x = LSTM(units, name='lstm')(x) # This has the effect of each LSTM unit returning a sequence of 1 output, one for each time step in the input data
# # x = Dense( n_output,name='lstm_out')(x)
# # x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='lstm_conv')(x)
# out = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)



print("======== lstm:")
lstm.summary()

lstm.compile(loss=custom_loss, optimizer=optimizer)

exit()
lstm.fit_generator(generator=train_batch,  # train_batch #(input, ground_truth)
                    steps_per_epoch=len(train_batch),
                    epochs=3,
                    verbose=1,
                    validation_data=valid_batch,
                    validation_steps=len(valid_batch),
                    callbacks=[early_stop, checkpoint, tensorboard],
                    max_queue_size=3)




"""evaluating on LSTM YOLO """
LSTM = evaluate(model, tohar_eval_batch, save_path=PATH+"/LSTM")
print("LSTM:\n",LSTM)
print(np.average(list(LSTM.values())))

# """evaluating on original YOLO (no training at all)"""
# model.load_weights("yolo.h5")
# YOLO = evaluate(model, tohar_eval_batch, save_path=PATH+"/YOLO")
# print("YOLO:\n", YOLO)
# print(np.average(list(YOLO.values())))
#
#
# """evaluating on original YOLO (no training at all) """
# model_t.load_weights(PATH+"/T_weights_coco.h5")
# NO_AE = evaluate(model_t, tohar_eval_batch, save_path=PATH+"/NO_AE")
# print("NO_AE:\n", NO_AE)
# print(np.average(list(NO_AE.values())))

params={"SUP_NUM_IMAGES:": SUP_NUM_IMAGES,
"UNSUP_NUM_IMAGES:":UNSUP_NUM_IMAGES,
"EVAL_NUM_IMAGES:":EVAL_NUM_IMAGES}

f = open(PATH + "/mAP.txt", "w")
f.write("LSTM:\n")
f.write(str(LSTM)+"\n")
f.write("NO_AE:\n")
# f.write(str(NO_AE)+"\n")
f.write("YOLO:\n")
# f.write(str(YOLO)+"\n")
f.write("AVG:"+"\n")
f.write(str(np.average(list(LSTM.values())))+"\n")
# f.write(str(np.average(list(NO_AE.values())))+"\n")
# f.write(str(np.average(list(YOLO.values())))+"\n")
f.write("LOG:"+"\n")
f.write(str(params) )


f.close()

# image = cv2.imread('images/giraffe.jpg')
# dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
# plt.figure(figsize=(10,10))
#
# input_image = cv2.resize(image, (416, 416))
# input_image = input_image / 255.
# input_image = input_image[:,:,::-1]
# input_image = np.expand_dims(input_image, 0)
#
# netout = model.predict([input_image, dummy_array])
#
# boxes = decode_netout(netout[0],
#                       obj_threshold=OBJ_THRESHOLD,
#                       nms_threshold=NMS_THRESHOLD,
#                       anchors=ANCHORS,
#                       nb_class=CLASS)
#
# image = draw_boxes(image, boxes, labels=LABELS)
#
# plt.imshow(image[:,:,::-1]); #plt.show()
# i=0
# plt.savefig("./predictions/figure"+str(i))
print('\a')
print('\a')

exit()













# # Perform detection on video

# In[ ]:


model.load_weights("weights_coco.h5")

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
