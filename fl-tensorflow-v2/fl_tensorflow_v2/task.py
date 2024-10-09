"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os

import tensorflow as tf
from flwr_datasets import FederatedDataset
from sklearn.model_selection import train_test_split

import pandas as pd

X_TEST = "../../UCI_HAR_Dataset/test/X_test.txt"
Y_TEST = "../../UCI_HAR_Dataset/test/y_test.txt"
X_TRAIN = "../../UCI_HAR_Dataset/train/X_train.txt"
Y_TRAIN = "../../UCI_HAR_Dataset/train/y_train.txt"
PATHCOL = "../../UCI_HAR_Dataset/features.txt"

USER_TEST = "../../UCI_HAR_Dataset/test/subject_test.txt"
USER_TRAIN = "../../UCI_HAR_Dataset/train/subject_train.txt"


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_model():
    # Load model and data (MobileNetV2, CIFAR-10)
    # model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(50, input_shape=(561, ), activation='relu'),
        # tf.keras.layers.Dense(6, activation='softmax')
        tf.keras.layers.Dense(50, input_shape=(561, )),
        tf.keras.layers.Dense(6)
    ])
    model.compile(
        "adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

def get_data(id):
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

    # get the user data
    ID_DF = DF[DF[0] == id]

    # return X, Y
    return ID_DF.iloc[:, 1:562], ID_DF.iloc[:, 562]

def load_data(partition_id, num_partitions):
    X, Y = get_data(partition_id)
    x_train, y_train, x_test, y_test = train_test_split(X, Y, random_state=42, test_size=0.3)

    return x_train, y_train, x_test, y_test


# this file has data & model so 
#  - i've to modify the data in order to change with the user data                          | DONE (to test)
#  - i've to modify the model in order to use the same model as the torch file (repo)       | DONE (to test)
#  - FEMNIST???                                                                             |