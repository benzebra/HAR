import io
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 0=None, 1=Vertical, 2=Horizontal
HYB_STATUS = 0
HAR = 0
HYB_PERCENTAGE = 0.5

DF_FEM = pd.read_parquet("hf://datasets/flwrlabs/femnist/data/train-00000-of-00001.parquet")
DF_WRITERS = DF_FEM["writer_id"].unique()

featmaps = [32, 64, 128]                                        # Example values for feature maps (number of filters)
kernels = [3, 3, 3]                                             # Example kernel sizes
first_linear_size = featmaps[2] * kernels[2] * kernels[2]       # Example size after flattening (depends on input size and feature maps)
linears = [512, 256, 62]                                        # Example sizes of the fully connected layers
num_classes = 62   

users_to_fed = []                                             # Number of classes in FEMNIST

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

    if(HYB_STATUS == 1):
        # array randomico con val da 0 a max con len max*HYB_PERCENTAGE
        max_users = get_max()
        users_to_fed = np.random.choice(max_users, int(max_users*HYB_PERCENTAGE), replace=False)
    return model

def load_data(partition_id, num_partitions):
    i = num_partitions-partition_id

    if(HYB_STATUS == 0):
        X, Y = get_data(i)

    elif(HYB_STATUS == 1):
        if(i == 1):
            others = np.setdiff1d(np.arange(get_max()), users_to_fed)
            if(HAR==0):
                X, Y = get_data(0)
                for n in range(len(others)):
                    X_tmp, Y_tmp = get_data(others[n])
                    X = pd.concat([X, X_tmp], ignore_index=False)
                    Y = pd.concat([Y, Y_tmp], ignore_index=False)
            else:
                _, Y = get_data(0)
                Y = Y.drop(Y.index)
                X = np.empty((0, 28, 28, 1))
                for n in range(len(others)):
                    X_tmp, Y_tmp = get_data(others[n])
                    X = np.concatenate((X, X_tmp), axis=0)
                    Y = pd.concat([Y, Y_tmp], ignore_index=False)
        else:
            i = i-1
            X, Y = get_data(users_to_fed[i])

    elif(HYB_STATUS == 2):
        if(i == 1):            
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

            X, Y = get_data(i)
            perc = int(len(X)-len(X)*HYB_PERCENTAGE)
            X_tmp = X[:perc]
            Y_tmp = Y[:perc]

            X = X_tmp
            Y = Y_tmp
                
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)
        
    return train_dataset, test_dataset






def get_data(id):
    if(HAR==0):
        ID_DF = DF_HAR[DF_HAR[0] == id]
        if(HYB_STATUS==2):
            ID_DF = ID_DF.sample(frac=1).reset_index(drop=True)
        X = ID_DF.iloc[:, 2:562]
        Y = ID_DF.iloc[:, 1] 

        if(HYB_STATUS == 2):
            X = X.shuffle(frac=1, random_state=42)
            Y = Y.shuffle(frac=1)
    else:
        ID_DF_FEM = DF_FEM[DF_FEM["writer_id"] == DF_WRITERS[id]]
        if(HYB_STATUS==2):
            ID_DF_FEM = ID_DF_FEM.sample(frac=1).reset_index(drop=True)
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