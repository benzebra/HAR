import flwr as fl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# folder dataset
PATH_TRAIN_X = "../UCI_HAR_Dataset/train/X_train.txt"
PATH_TRAIN_Y = "../UCI_HAR_Dataset/train/y_train.txt"

PATH_TEST_X = "../UCI_HAR_Dataset/test/X_test.txt"
PATH_TEST_Y = "../UCI_HAR_Dataset/test/y_test.txt"

PATH_TRAIN_SBJ = "../UCI_HAR_Dataset/train/subject_train.txt"
PATH_TEST_SBJ = "../UCI_HAR_Dataset/test/subject_test.txt"
PATH_FT = "../UCI_HAR_Dataset/features.txt"

# read the dataset as pandas.Dataframe
# features 
features = pd.read_csv(PATH_FT, sep=" ", header=None, index_col=0).reset_index()
# training X
df_x_train = pd.read_fwf(PATH_TRAIN_X, header=None)
df_x_train.rename(columns=features[1], inplace=True)
# activity subject
df_sbj_train = pd.read_fwf(PATH_TRAIN_SBJ, header=None)
df_x_train['user'] = df_sbj_train.values
# training Y
y_train_col = pd.read_fwf(PATH_TRAIN_Y, header=None)
# convert to np.array
X = np.array(df_x_train)
y = np.array(y_train_col)

# TODO: add the subject filter (in: sbj number)

# splitting and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class UCIHARClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": float(accuracy)}
    
fl.client.start_client(server_address="[::]:8080", client=UCIHARClient().to_client())