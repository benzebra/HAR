"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os
from flwr.client import NumPyClient, ClientApp
from fl_tensorflow_v2.task import load_data, load_model
# from fl_tensorflow_v2.task import USERS, EPOCHS

USERS = 10
EPOCHS = 10

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    # def __init__(self, model, x_train, x_test, y_train, y_test):
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_test = x_test
        # self.y_test = y_test
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # print("debug client fit")
        self.model.set_weights(parameters)
        # print("debug client set weights")
        self.model.fit(self.train_dataset, epochs=EPOCHS, batch_size=32, verbose=0)
        # self.model.fit(self.x_train, self.y_train, epochs=EPOCHS, batch_size=32, verbose=0)
        # print("debug client fited ")
        # return self.model.get_weights(), len(self.x_train), {}
        return self.model.get_weights(), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        loss, accuracy = self.model.evaluate(self.test_dataset)
        # print(len(self.x_test))
        # float, int, {srt: float}
        # return loss, len(self.x_test), {"accuracy": float(accuracy)}
        return loss, len(self.test_dataset), {"accuracy": float(accuracy)}       # f1 score


def client_fn(cid):
    net = load_model()
    # x_train, x_test, y_train, y_test = load_data(int(cid), USERS)
    train_dataset, test_dataset = load_data(int(cid), USERS)

    # Return Client instance
    # return FlowerClient(net, x_train, x_test, y_train, y_test).to_client()
    return FlowerClient(net, train_dataset, test_dataset).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Add f1 score in the evaluation returns        |
# Adjust the number of the epochs               | DONE  
# FEMNIST ???                                   |
# Check cid (maybe it starts from 0)            | DONE