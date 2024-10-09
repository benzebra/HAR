"""FL-tensorflow-v2: A Flower / TensorFlow app."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from fl_tensorflow_v2.task import load_model

# Define config
# print("ServerApp: Defining config")
config = ServerConfig(num_rounds=3, round_timeout=300)
# print(f"ServerApp: Config: {config}")

# print("ServerApp: Loading model")
parameters = ndarrays_to_parameters(load_model().get_weights())
# print(f"ServerApp: Tensor Type: {parameters.tensor_type}")

# print("ServerApp: Defining strategy")
# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=3,
    initial_parameters=parameters,
)
# print(f"ServerApp: strategy: {strategy}")

# print("ServerApp: Creating ServerApp")  
# Create ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)
# print(f"ServerApp: ServerApp: {app}")
