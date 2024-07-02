import flwr as fl

# fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))
from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
# strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,                   # Sample 10% of available clients for training
    fraction_evaluate=0.05,             # Sample 5% of available clients for evaluation
    min_fit_clients=3,                  # Never sample less than 10 clients for training
    min_evaluate_clients=3,             # Never sample less than 5 clients for evaluation
    min_available_clients=int(
        3
    ),                                  # Wait until at least 75 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
    # evaluate_fn=get_evaluate_fn(centralized_testset),  # global evaluation function
)

# Define config
config = ServerConfig(num_rounds=3)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )