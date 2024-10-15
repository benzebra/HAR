"""FL-tensorflow-v2: A Flower / TensorFlow app."""
# import numpy as np

# USERS = 46
# ROUNDS = 1
# MAX = 50
# HYB_PERCENTAGE = 0.1

# users_to_fed = np.random.choice(MAX, int(MAX*HYB_PERCENTAGE), replace=False)
# print(f"users_to_fed: {users_to_fed}")

import os
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from typing import List, Tuple

USERS = 31
ROUNDS = 1

from fl_tensorflow_v2.task import load_model

# Define config
config = ServerConfig(num_rounds=ROUNDS)

parameters = ndarrays_to_parameters(load_model().get_weights())

def simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    s_accuracies = [m["accuracy"] for _, m in metrics]
    clients_num = len(metrics)
    accuracy = sum(s_accuracies) / clients_num

    print_values(accuracy)

    return {"accuracy": accuracy}

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=USERS,
    initial_parameters=parameters,
    evaluate_metrics_aggregation_fn=simple_average,
)

app = ServerApp(
    config=config,
    strategy=strategy,
)

def print_values(accuracy):
    with open("metrics_har_hor_10t.txt", "a") as file:
        file.write(f"{accuracy},")
    return