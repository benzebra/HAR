
import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from typing import Dict

from flwr_datasets import FederatedDataset

PATH_TRAIN_X = "../UCI_HAR_Dataset/train/X_train.txt"
PATH_TRAIN_Y = "../UCI_HAR_Dataset/train/y_train.txt"

PATH_TEST_X = "../UCI_HAR_Dataset/test/X_test.txt"
PATH_TEST_Y = "../UCI_HAR_Dataset/test/y_test.txt"

PATH_TRAIN_SBJ = "../UCI_HAR_Dataset/train/subject_train.txt"

PATH_TEST_SBJ = "../UCI_HAR_Dataset/test/subject_test.txt"

PATH_FT = "../UCI_HAR_Dataset/features.txt"
features = pd.read_csv(PATH_FT, sep=" ", header=None, index_col=0).reset_index()

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # # Load test data here to avoid the overhead of doing it in `evaluate` itself
    # fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    # dataset = fds.load_split("test").with_format("numpy")
    # X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"]

    # Load the partition data
    df_x_train = pd.read_fwf(PATH_TRAIN_X, header=None)
    df_x_train.rename(columns=features[1], inplace=True)

    y_train_col = pd.read_fwf(PATH_TRAIN_Y, header=None)
    print(f"x_train shape: {df_x_train.shape}\ny_train shape: {y_train_col.shape}")
    
    # SPLITTING
    X_train, X_test, y_train, y_test = train_test_split(df_x_train, y_train_col, random_state=42, test_size=0.3)
    print(f"X_train shape: {X_train.shape}\nX_test shape: {X_test.shape}\ny_train shape: {y_train.shape}\ny_test shape: {y_test.shape}")

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    # model = LogisticRegression()
    model = RandomForestClassifier()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
