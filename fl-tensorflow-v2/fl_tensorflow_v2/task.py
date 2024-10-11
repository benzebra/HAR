"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os

import tensorflow as tf
from flwr_datasets import FederatedDataset
from sklearn.model_selection import train_test_split

import pandas as pd

# !!!
# When i run task.py it goes to the main directory so these are the correct paths
X_TEST = "../UCI_HAR_Dataset/test/X_test.txt"
Y_TEST = "../UCI_HAR_Dataset/test/y_test.txt"
X_TRAIN = "../UCI_HAR_Dataset/train/X_train.txt"
Y_TRAIN = "../UCI_HAR_Dataset/train/y_train.txt"
PATHCOL = "../UCI_HAR_Dataset/features.txt"
USER_TEST = "../UCI_HAR_Dataset/test/subject_test.txt"
USER_TRAIN = "../UCI_HAR_Dataset/train/subject_train.txt"

# load df
DF_X_TEST = pd.read_fwf(X_TEST, header=None)
DF_Y_TEST = pd.read_fwf(Y_TEST, header=None)
DF_X_TRAIN = pd.read_fwf(X_TRAIN, header=None)
DF_Y_TRAIN = pd.read_fwf(Y_TRAIN, header=None)
DF_PATHCOL = pd.read_fwf(PATHCOL, header=None)
DF_USER_TEST = pd.read_fwf(USER_TEST, header=None, widths=[2])
DF_USER_TRAIN = pd.read_fwf(USER_TRAIN, header=None, widths=[2])

DF_TEST = pd.concat([DF_USER_TEST, DF_Y_TEST, DF_X_TEST], axis=1, ignore_index=True)
DF_TRAIN = pd.concat([DF_USER_TRAIN, DF_Y_TRAIN, DF_X_TRAIN], axis=1, ignore_index=True)
DF = pd.concat([DF_TRAIN, DF_TEST], axis=0, ignore_index=True)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# 0=None, 1=Vertical, 2=Horizontal
HYB_STATUS = 1
HYB_PERCENTAGE = 0.5


def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(560, )),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def load_data(partition_id, num_partitions):
    i = num_partitions-partition_id

    if(HYB_STATUS == 0):
        X, Y = get_data(i)

    elif(HYB_STATUS == 1):
        if(i == 1):
            X, Y = get_data(0)
            usr_out = max(DF[0])
            for n in range(num_partitions, usr_out+1):
                print(f"n: {n}")
                X_tmp, Y_tmp = get_data(n)

                X = pd.concat([X, X_tmp], ignore_index=False)
                Y = pd.concat([Y, Y_tmp], ignore_index=False)
            print(f"X: {len(X)}")
        else:
            i = i-1
            print(f"i: {i}")
            X, Y = get_data(i)


    elif(HYB_STATUS == 2):
        if(i == 1):
            X, Y = get_data(0)
            for n in range(1, num_partitions):
                X_tmp, Y_tmp = get_data(n)

                perc = int(len(X_tmp)-len(X_tmp)*HYB_PERCENTAGE)

                X = pd.concat([X, X_tmp[perc:]], ignore_index=False)
                Y = pd.concat([Y, Y_tmp[perc:]], ignore_index=False)
        else: 
            i = i-1
            X, Y = get_data(i)

            perc = int(len(X)-len(X)*HYB_PERCENTAGE)

            X_tmp = X[:perc]
            Y_tmp = Y[:perc]

            X = X_tmp
            Y = Y_tmp
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3)
    return x_train, x_test, y_train, y_test

def get_data(id):
    ID_DF = DF[DF[0] == id]

    return ID_DF.iloc[:, 2:562], ID_DF.iloc[:, 1]


# this file has data & model so 
#  - i've to modify the data in order to change with the user data                          | DONE (to test)
#  - i've to modify the model in order to use the same model as the torch file (repo)       | DONE (to test)
#  - FEMNIST???                                                                             |