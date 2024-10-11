"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os
from flwr.client import NumPyClient, ClientApp
from fl_tensorflow_v2.task import load_data, load_model

USERS = 16
EPOCHS = 10

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=EPOCHS, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        # print(len(self.x_test))
        # float, int, {srt: float}
        return loss, len(self.x_test), {"accuracy": float(accuracy)}       # f1 score


def client_fn(cid):
    net = load_model()
    x_train, x_test, y_train, y_test = load_data(int(cid), USERS)

    # Return Client instance
    return FlowerClient(net, x_train, x_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Add f1 score in the evaluation returns        |
# Adjust the number of the epochs               |   
# FEMNIST ???                                   |
# Check cid (maybe it starts from 0)            | DONE