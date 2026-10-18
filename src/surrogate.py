"""Surrogate models that predict the oracle's output *with uncertainty*.

Uncertainty is the fuel of active learning: it tells the agent where the
model is guessing, so it can choose between exploring the unknown and
exploiting what looks good. Two interchangeable surrogates are provided so
the agent can *orchestrate* between them:

- RandomForest: robust, no feature scaling, uncertainty from tree spread.
- GaussianProcess: smooth, well-calibrated std, better in the low-data regime.

Both expose the same `.fit(X, y)` / `.predict(X) -> (mean, std)` interface.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class RFSurrogate:
    """Random-forest surrogate; std = dispersion across the individual trees."""

    name = "random_forest"

    def __init__(self, n_estimators: int = 300, seed: int = 0) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFSurrogate":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)
        per_tree = np.stack([t.predict(X) for t in self.model.estimators_], axis=0)
        mean = per_tree.mean(axis=0)
        std = per_tree.std(axis=0) + 1e-9
        return mean, std


class GPSurrogate:
    """Gaussian-process surrogate with input standardisation."""

    name = "gaussian_process"

    def __init__(self, seed: int = 0) -> None:
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * RBF(length_scale=np.ones(1), length_scale_bounds=(1e-1, 1e2))
            + WhiteKernel(1.0, (1e-3, 1e1))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed,
        )
        self.x_scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        Xs = self.x_scaler.fit_transform(X)
        # One shared length scale, re-expanded to the feature dimension.
        self.model.kernel.k1.k2.length_scale = np.ones(X.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = self.x_scaler.transform(np.atleast_2d(X))
        mean, std = self.model.predict(Xs, return_std=True)
        return mean, std + 1e-9


SURROGATES = {"random_forest": RFSurrogate, "gaussian_process": GPSurrogate}


def make_surrogate(name: str, seed: int = 0):
    if name not in SURROGATES:
        raise ValueError(f"unknown surrogate {name!r}; choose from {list(SURROGATES)}")
    return SURROGATES[name](seed=seed)
