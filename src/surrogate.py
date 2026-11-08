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

    def fit(self, X: np.ndarray, y: np.ndarray,
            weights: np.ndarray | None = None) -> RFSurrogate:
        self.model.fit(X, y, sample_weight=weights)
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

    def fit(self, X: np.ndarray, y: np.ndarray,
            weights: np.ndarray | None = None) -> GPSurrogate:
        # A GP has no sample_weight, but it has per-point noise: a measurement
        # trusted less is a measurement with more noise on it, which is the same
        # statement. alpha scales as 1/weight, so a low-fidelity reading pulls
        # the posterior less without being discarded.
        if weights is not None and len(weights):
            w = np.clip(np.asarray(weights, dtype=float), 1e-3, None)
            self.model.alpha = float(np.var(y) or 1.0) * (1.0 / w) * 1e-2
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


class EnsembleMLPSurrogate:
    """An ensemble of small neural nets; uncertainty from disagreement.

    The GP is the right model for twenty points in six dimensions and the wrong
    one for ten thousand points over a 1024-bit fingerprint -- it is cubic in
    the number of observations and assumes a smooth stationary kernel that
    fingerprint space does not have. This is the surrogate for that regime, and
    it is what the screening literature actually uses: a handful of independently
    initialised networks, with the spread of their predictions standing in for a
    posterior.
    """

    name = "ensemble_mlp"

    def __init__(self, n_models: int = 5, hidden: tuple[int, ...] = (128, 64),
                 seed: int = 0, max_iter: int = 300) -> None:
        from sklearn.neural_network import MLPRegressor

        self.models = [
            MLPRegressor(hidden_layer_sizes=hidden, random_state=seed + i,
                         max_iter=max_iter, early_stopping=False, alpha=1e-3)
            for i in range(n_models)
        ]
        self.x_scaler = StandardScaler()
        self._rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray,
            weights: np.ndarray | None = None) -> EnsembleMLPSurrogate:
        Xs = self.x_scaler.fit_transform(X)
        n = len(y)
        prob = None
        if weights is not None and len(weights):
            w = np.clip(np.asarray(weights, dtype=float), 1e-6, None)
            prob = w / w.sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for model in self.models:
                # Weighted bootstrap: the ensemble spread *is* the uncertainty,
                # so each member must see a different resample of the data.
                idx = self._rng.choice(n, size=n, replace=True, p=prob)
                model.fit(Xs[idx], np.asarray(y)[idx])
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = self.x_scaler.transform(np.atleast_2d(X))
        preds = np.stack([m.predict(Xs) for m in self.models], axis=0)
        return preds.mean(axis=0), preds.std(axis=0) + 1e-9


SURROGATES = {
    "random_forest": RFSurrogate,
    "gaussian_process": GPSurrogate,
    "ensemble_mlp": EnsembleMLPSurrogate,
}


def make_surrogate(name: str, seed: int = 0):
    if name not in SURROGATES:
        raise ValueError(f"unknown surrogate {name!r}; choose from {list(SURROGATES)}")
    return SURROGATES[name](seed=seed)
