"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os

import tensorflow as tf
from flwr_datasets import FederatedDataset


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_model():
    # Load model and data (MobileNetV2, CIFAR-10)
    # model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, input_shape=(561, ), activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(
        "adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    return x_train, y_train, x_test, y_test

# this file has data & model so 
#  - i've to modify the data in order to change with the user data                          |
#  - i've to modify the model in order to use the same model as the torch file (repo)       | DONE

#  - FEMNIST???                                                                             |