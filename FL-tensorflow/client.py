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

N_CLIENTS = 30

# folder dataset
PATH_TRAIN_X = "../UCI_HAR_Dataset/train/X_train.txt"
PATH_TRAIN_Y = "../UCI_HAR_Dataset/train/y_train.txt"

PATH_TEST_X = "../UCI_HAR_Dataset/test/X_test.txt"
PATH_TEST_Y = "../UCI_HAR_Dataset/test/y_test.txt"

PATH_TRAIN_SBJ = "../UCI_HAR_Dataset/train/subject_train.txt"
PATH_TEST_SBJ = "../UCI_HAR_Dataset/test/subject_test.txt"
PATH_FT = "../UCI_HAR_Dataset/features.txt"

# model definition
model = tf.keras.Sequential([
    # layers.InputLayer(shape=(X.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# read the dataset as pandas.Dataframe
# features 
features = pd.read_csv(PATH_FT, sep=" ", header=None, index_col=0).reset_index()

# training X
df_x_train = pd.read_fwf(PATH_TRAIN_X, header=None)
df_x_train.rename(columns=features[1], inplace=True)
# activity subject (train)
df_sbj_train = pd.read_csv(PATH_TRAIN_SBJ, sep=" ", header=None)
df_x_train['user'] = df_sbj_train.values
arr_sbj_train = (df_sbj_train.iloc[:,0]).to_list()
# training Y
y_train_col = pd.read_fwf(PATH_TRAIN_Y, header=None)

# testing X
df_x_test = pd.read_fwf(PATH_TEST_X, header=None)
df_x_test.rename(columns=features[1], inplace=True)
# activity subject (test)
df_sbj_test = pd.read_csv(PATH_TEST_SBJ, sep=" ", header=None)
df_x_test['user'] = df_sbj_test.values
arr_sbj_test = (df_sbj_test.iloc[:,0]).to_list()
# testing Y
y_test_col = pd.read_fwf(PATH_TEST_Y, header=None)

# get the user data function, it returns [ X_train, y_train, X_test, y_test ] 
def getData(user):
    usr_act_train = []                              # activity made by user (train)
    usr_act_test = []                               # activity made by user (test)

    for i in range(len(arr_sbj_train)):
        if(arr_sbj_train[i] == user):
            usr_act_train.append(i)
    
    for i in range(len(arr_sbj_test)):
        if(arr_sbj_test[i] == user):
            usr_act_test.append(i)
    
    df_ext_train = pd.DataFrame(dtype=float)
    df_ext_test = pd.DataFrame(dtype=float)
    y_ext_train = pd.DataFrame(dtype=float)
    y_ext_test = pd.DataFrame(dtype=float)

    for i in range(len(usr_act_train)):
        index = usr_act_train[i]
        x_row = df_x_train.iloc[index,:561]
        df_ext_train = pd.concat([df_ext_train, x_row], ignore_index=True, axis=1)
        y_row = y_train_col.iloc[index]
        y_ext_train = pd.concat([y_ext_train, y_row], ignore_index=True, axis=1)

    for i in range(len(usr_act_test)):
        index = usr_act_test[i]
        x_row = df_x_test.iloc[index,:561]
        df_ext_test = pd.concat([df_ext_test, x_row], ignore_index=True, axis=1)
        y_row = y_test_col.iloc[index]
        y_ext_test = pd.concat([y_ext_test, y_row], ignore_index=True, axis=1)

    df_ext_train = df_ext_train.T
    y_ext_train = y_ext_train.T
    X_train = np.array(df_ext_train)
    y_train = np.array(y_ext_train)

    df_ext_test = df_ext_test.T
    y_ext_test = y_ext_test.T
    X_test = np.array(df_ext_test)
    y_test = np.array(y_ext_test)

    # TODO: 
    # if test.shape = [0,0] and train.shape != [0,0] use the train_test_split function
    # train_test_split(X, y, random_state=42, test_size=0.3)
    # 
    # if train.shape = [0,0] but test.shape != [0,0] ... ???

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":

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

    X_train, y_train, X_test, y_test = getData(user)

    print(f"train shape: X-" + str(X_train.shape) + " y-" + str(y_train.shape))
    print(f"test shape: X-" + str(X_test.shape) + " y-" + str(y_test.shape))

#     utils.set_initial_params(model)   


#     class UCIHARClient(fl.client.NumPyClient):
#         def get_parameters(self, config):
#             return model.get_weights()

#         def fit(self, parameters, config):
#             model.set_weights(parameters)
#             # model.fit(X_user_train, y_user_train, epochs=1, batch_size=32, steps_per_epoch=3)
#             model.fit(X_train, y_train, epochs=10, batch_size=32)
#             return model.get_weights(), len(X_train), {}

#         def evaluate(self, parameters, config):
#             model.set_weights(parameters)
#             loss, accuracy = model.evaluate(X_test, y_test)
#             return loss, len(X_test), {"accuracy": float(accuracy)}
    

# # start the client
# fl.client.start_client(server_address="[::]:8080", client=UCIHARClient().to_client())