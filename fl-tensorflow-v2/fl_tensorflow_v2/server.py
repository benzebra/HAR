"""FL-tensorflow-v2: A Flower / TensorFlow app."""

import os
from flwr.common import ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from typing import List, Tuple

from fl_tensorflow_v2.task import load_model

USERS = 30
ROUNDS = 1

# Define config
# print("ServerApp: Defining config")
config = ServerConfig(num_rounds=ROUNDS)
# print(f"ServerApp: Config: {config}")

# print("ServerApp: Loading model")
parameters = ndarrays_to_parameters(load_model().get_weights())
# print(f"ServerApp: Tensor Type: {parameters.tensor_type}")

# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     print(f"ServerApp: Accuracy type: {type(accuracies)}")
#     return {"accuracy": sum(accuracies) / sum(examples)}

def simple_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # print(metrics)
    # log(DEBUG, f"current metrics: {metrics}")
    s_accuracies = [m["accuracy"] for _, m in metrics]
    clients_num = len(metrics)
    # log(DEBUG, f"NUMBER CLIENTS {clients_num}")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(s_accuracies)/clients_num}

# print("ServerApp: Defining strategy")
# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=USERS,
    initial_parameters=parameters,
    evaluate_metrics_aggregation_fn=simple_average,
    # evaluate_metrics_aggregation_fn=weighted_average,
)
# print(f"ServerApp: strategy: {strategy}")

# print("ServerApp: Creating ServerApp")  
# Create ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)
# print(f"ServerApp: ServerApp: {app}")
