"""Diversity-aware batch selection.

Taking the top-k by acquisition score is the obvious thing and the wrong thing.
Acquisition functions are smooth in feature space, so the k highest scorers are
usually k near-identical molecules from the same chemical series. A batch of six
analogues costs six docking runs and returns roughly one bit of information: if
the first one fails, so will the other five.

This matters more in screening than in continuous optimisation, because a
library contains dense clusters of close analogues by construction -- that is
how combinatorial catalogues are built.

The fix is the standard one: pick greedily by acquisition score, but refuse a
candidate that sits too close to something already in the batch. `lambda_div`
sets how much score you are willing to give up for separation.
"""

from __future__ import annotations

import numpy as np


def greedy_maxmin(features: np.ndarray, scores: np.ndarray, k: int,
                  lambda_div: float = 0.5) -> np.ndarray:
    """Select k indices trading acquisition score against distance from the batch.

    The first pick is the highest scorer. Each later pick maximises

        normalised_score - lambda_div * normalised_similarity_to_batch

    With `lambda_div = 0` this is exactly top-k, which keeps the old behaviour
    one argument away and makes the comparison between them a clean experiment.
    """
    features = np.atleast_2d(features)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)
    if lambda_div <= 0 or k == 1:
        return np.argsort(scores)[::-1][:k]

    span = scores.max() - scores.min()
    norm_scores = (scores - scores.min()) / span if span > 1e-12 else np.zeros_like(scores)

    # Scale distances so lambda_div means the same thing whatever the
    # featurisation. The scale has to be a *small* percentile of the pairwise
    # distances, not the median: in a library full of analogue clusters the
    # median is set by how far apart the clusters are, which makes every pair
    # inside a cluster look identically close and kills the diversity term
    # exactly where it was needed.
    sample = features[np.linspace(0, n - 1, min(n, 256)).astype(int)]
    pair = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    off_diagonal = pair[np.triu_indices(len(sample), k=1)]
    typical = float(np.percentile(off_diagonal, 10)) if len(off_diagonal) else 1.0
    if typical <= 1e-12:
        typical = float(off_diagonal.mean()) if len(off_diagonal) else 1.0
    typical = typical or 1.0

    chosen = [int(np.argmax(norm_scores))]
    min_dist = np.linalg.norm(features - features[chosen[0]], axis=1) / typical

    for _ in range(k - 1):
        # Closeness in [0, 1]: 1 means sitting on top of an already-chosen point.
        closeness = np.exp(-min_dist)
        utility = norm_scores - lambda_div * closeness
        utility[chosen] = -np.inf
        pick = int(np.argmax(utility))
        chosen.append(pick)
        min_dist = np.minimum(min_dist, np.linalg.norm(features - features[pick], axis=1) / typical)

    return np.array(chosen, dtype=int)


def batch_diversity(features: np.ndarray, indices: np.ndarray) -> float:
    """Mean pairwise distance inside a batch -- a one-number diversity report."""
    pts = np.atleast_2d(features)[np.asarray(indices, dtype=int)]
    if len(pts) < 2:
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(len(pts), k=1)
    return float(d[iu].mean())
