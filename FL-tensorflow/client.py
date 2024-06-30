import flwr as fl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import argparse
import warnings
import utils

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

# splitting and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    layers.InputLayer(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

if __name__ == "__main__":
    N_CLIENTS = 30

    # ---------------
    # get the user input
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--user",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the user (--user)",
    )
    args = parser.parse_args()
    user = args.user
    # ---------------

    # utils.set_initial_params(model)

    # ---------------
    # user splitting
    df_sbj = pd.read_fwf(PATH_TRAIN_SBJ, header=None)

    user_index = user                       # I want to know user-th infos
    usr_act = []                            # activity made by user
    arr_sbj = (df_sbj.iloc[:,0]).to_list()

    for i in range(len(arr_sbj)):
        if(arr_sbj[i] == user_index):
            usr_act.append(i)
    
    df_ext = pd.DataFrame(dtype=float)
    y_ext = pd.DataFrame(dtype=float)

    for i in range(len(usr_act)):
        index = usr_act[i]
        new_row_x = df_x_train.iloc[index]
        df_ext = pd.concat([df_ext, new_row_x], ignore_index=True, axis=1)
        new_row_y = y_train_col.iloc[index]
        y_ext = pd.concat([y_ext, new_row_y], ignore_index=True, axis=1)

    df_ext = df_ext.T
    y_ext = y_ext.T
    X = np.array(df_ext)
    y = np.array(y_ext)

    if(len(df_ext) != 0):
        X_user_train, X_user_test, y_user_train, y_user_test = train_test_split(X, y, random_state=42, test_size=0.3)
    else:
        X_user_train, X_user_test, y_user_train, y_user_test = []
    # ----------------
                          
#     class UCIHARClient(fl.client.NumPyClient):
#         def get_parameters(self, config):
#             return model.get_weights()

#         def fit(self, parameters, config):
#             model.set_weights(parameters)
#             model.fit(X_user_train, y_user_train, epochs=1, batch_size=32, steps_per_epoch=3)
#             return model.get_weights(), len(X_user_train), {}

#         def evaluate(self, parameters, config):
#             model.set_weights(parameters)
#             loss, accuracy = model.evaluate(X_user_test, y_user_test)
#             return loss, len(X_user_test), {"accuracy": float(accuracy)}
    

# # start the client
# fl.client.start_client(server_address="[::]:8080", client=UCIHARClient().to_client())