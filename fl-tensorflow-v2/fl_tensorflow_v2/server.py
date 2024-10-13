"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from typing import List, Tuple

from fl_tensorflow_v2.task import load_model

# # 0=None, 1=Vertical, 2=Horizontal
# HYB_STATUS = 0

# HYB_PERCENTAGE = 0.5
USERS = 10
ROUNDS = 1
# EPOCHS = 10

# Define config
config = ServerConfig(num_rounds=ROUNDS)

parameters = ndarrays_to_parameters(load_model().get_weights())

def simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    s_accuracies = [m["accuracy"] for _, m in metrics]
    clients_num = len(metrics)

    print_values()

    return {"accuracy": sum(s_accuracies)/clients_num}

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

def print_values():
    return