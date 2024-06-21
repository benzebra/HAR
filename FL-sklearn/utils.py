import numpy as np
from sklearn.ensemble import RandomForestClassifier

from flwr.common import NDArrays


def get_model_parameters(model: RandomForestClassifier) -> NDArrays:
    """Returns the parameters of a sklearn RandomForestClassifier model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: RandomForestClassifier, params: NDArrays) -> RandomForestClassifier:
    """Sets the parameters of a sklean RandomForestClassifier model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: RandomForestClassifier):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 6
    n_features = 561 
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))