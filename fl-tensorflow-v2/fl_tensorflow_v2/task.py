"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os

import tensorflow as tf
from flwr_datasets import FederatedDataset
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import io
# from fl_tensorflow_v2.server import HYB_STATUS, HYB_PERCENTAGE

import pandas as pd

# load df
DF_X_TEST = pd.read_fwf("../UCI_HAR_Dataset/test/X_test.txt", header=None)
DF_Y_TEST = pd.read_fwf("../UCI_HAR_Dataset/test/y_test.txt", header=None)
DF_X_TRAIN = pd.read_fwf("../UCI_HAR_Dataset/train/X_train.txt", header=None)
DF_Y_TRAIN = pd.read_fwf("../UCI_HAR_Dataset/train/y_train.txt", header=None)
DF_PATHCOL = pd.read_fwf("../UCI_HAR_Dataset/features.txt", header=None)
DF_USER_TEST = pd.read_fwf("../UCI_HAR_Dataset/test/subject_test.txt", header=None, widths=[2])
DF_USER_TRAIN = pd.read_fwf("../UCI_HAR_Dataset/train/subject_train.txt", header=None, widths=[2])
DF_TEST = pd.concat([DF_USER_TEST, DF_Y_TEST, DF_X_TEST], axis=1, ignore_index=True)
DF_TRAIN = pd.concat([DF_USER_TRAIN, DF_Y_TRAIN, DF_X_TRAIN], axis=1, ignore_index=True)
DF_HAR = pd.concat([DF_TRAIN, DF_TEST], axis=0, ignore_index=True)

DF_FEM = pd.read_parquet("hf://datasets/flwrlabs/femnist/data/train-00000-of-00001.parquet")
DF_WRITERS = DF_FEM["writer_id"].unique()

# 0=None, 1=Vertical, 2=Horizontal
HYB_STATUS = 2
HYB_PERCENTAGE = 1.0

# 0=HAR, 1=FEMNIST
HAR = 1

featmaps = [32, 64, 128]                                        # Example values for feature maps (number of filters)
kernels = [3, 3, 3]                                             # Example kernel sizes
first_linear_size = featmaps[2] * kernels[2] * kernels[2]       # Example size after flattening (depends on input size and feature maps)
linears = [512, 256, 62]                                        # Example sizes of the fully connected layers
num_classes = 62                                                # Number of classes in FEMNIST

def load_model():
    if(HAR == 0):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(560, )),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=featmaps[0], kernel_size=kernels[0], padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'),
            tf.keras.layers.Conv2D(filters=featmaps[1], kernel_size=kernels[1], padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'),
            tf.keras.layers.Conv2D(filters=featmaps[2], kernel_size=kernels[2], padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(linears[0], activation='relu'),
            tf.keras.layers.Dense(linears[1]),
            tf.keras.layers.Dense(num_classes, activation='log_softmax'),
        ])

    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy", "f1_score"])

    return model

def load_data(partition_id, num_partitions):
    # print(f"load_data partition_id: {partition_id}")
    i = num_partitions-partition_id

    # print(f"load_data i: {i}")

    if(HYB_STATUS == 0):
        # print(f"debug task, i = {i}")
        X, Y = get_data(i)

    # to test
    elif(HYB_STATUS == 1):
        if(i == 1):
            if(HAR==0):
                X, Y = get_data(0)
                usr_out = get_max()
                for n in range(num_partitions, usr_out+1):
                    print(f"n: {n}")
                    X_tmp, Y_tmp = get_data(n)

                    X = pd.concat([X, X_tmp], ignore_index=False)
                    Y = pd.concat([Y, Y_tmp], ignore_index=False)
            else:
                _, Y = get_data(0)
                Y = Y.drop(Y.index)
                X = np.empty((0, 28, 28, 1))
                usr_out = get_max()
                for n in range(num_partitions-1, usr_out):
                    print(f"n: {n}")
                    X_tmp, Y_tmp = get_data(n)
                    X = np.concatenate((X, X_tmp), axis=0)
                    Y = pd.concat([Y, Y_tmp], ignore_index=False)
                print(len(X))
        else:
            if(HAR==0):
                i = i-1
                print(f"i: {i}")
            else:
                i = i-2
                print(f"i: {i}")
            X, Y = get_data(i)

    elif(HYB_STATUS == 2):
        # print(f"debug task, i = {i}")
        if(i == 1):
            # X, Y = get_data(0)
            # delete_data(X, Y)
            
            if(HAR==0):
                X, Y = get_data(0)
                j = 1
                stop = num_partitions
            else:
                j = 0
                stop = num_partitions-1
                _, Y = get_data(0)
                Y = Y.drop(Y.index)
                X = np.empty((0, 28, 28, 1))

            for n in range(j, stop):
                X_tmp, Y_tmp = get_data(n)

                perc = int(len(X_tmp)-len(X_tmp)*HYB_PERCENTAGE)

                if(HAR==0):
                    X = pd.concat([X, X_tmp[perc:]], ignore_index=False)
                else:
                    X = np.concatenate((X, X_tmp[perc:]), axis=0)
                Y = pd.concat([Y, Y_tmp[perc:]], ignore_index=False)
        else: 
            if(HAR==0):
                i = i-1
            else:
                i = i-2
            print(i)
            X, Y = get_data(i)

            perc = int(len(X)-len(X)*HYB_PERCENTAGE)

            X_tmp = X[:perc]
            Y_tmp = Y[:perc]

            X = X_tmp
            Y = Y_tmp
                
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)
        
    # return x_train, x_test, y_train, y_test
    return train_dataset, test_dataset

def get_data(id):
    if(HAR==0):
        ID_DF = DF_HAR[DF_HAR[0] == id]
        X = ID_DF.iloc[:, 2:562]
        Y = ID_DF.iloc[:, 1] 
    else:
        ID_DF_FEM = DF_FEM[DF_FEM["writer_id"] == DF_WRITERS[id]]
        images = []
        for img_dict in ID_DF_FEM['image']:
            img_bytes = img_dict['bytes']
            img = Image.open(io.BytesIO(img_bytes)).convert('L')    # 'L' per convertire in scala di grigi
            img_array = np.array(img).reshape(28, 28, 1)
            images.append(img_array)
        images = np.array(images).astype(np.float32) / 255.0        # Normalizza le immagini (da 0 a 1)
        X = images
        # print(f"get_data X: {len(X)}")
        # X=pd.DataFrame(images)
        # Y = ID_DF_FEM.iloc[:, 3].values.astype(np.int64)
        Y = ID_DF_FEM.iloc[:, 3]

    # print(type(X), type(Y))
    return X, Y

def get_max():
    if(HAR==0):
        return 30
        return max(DF_HAR[0])
    else:
        # taglia array 
        return 50
        return len(DF_WRITERS)