"""Synthetic-dataset loader: deterministic, contamination-free classification tasks.

Unlike the fetch-from-a-source loaders, this one *generates* its data from a fixed recipe
and seed, so the exact rows cannot appear in any model's pretraining corpus — an
uncontaminated anchor alongside the public OpenML datasets (a model that scores well on
the memorisable public tables but at chance here is exploiting memorisation, not
in-context learning).

The recipes span deliberately different **generative priors**, because "can an LLM do
in-context tabular classification" has no single answer: a model can recover an additive
smooth function and still fail on parity, and the hypothesis classes below are the ones
that separate the model families. Grouped by what they probe:

*Boundary geometry*
  ``linear``    binary; linear boundary over 8 features (~3% label noise).
  ``rings``     binary; radial boundary in 2 of 6 dims — non-monotone, not axis-aligned.
  ``blobs``     3-class; Gaussian clusters separated in 2 of 6 dims.

*Interaction order* (each is unlearnable by a model that only sees marginals)
  ``xor``       binary; XOR of two features amid 2 distractors.
  ``parity3``   binary; 3-way parity amid 5 distractors — strictly harder than ``xor``.
  ``checkers``  binary; 4x4 checkerboard in 2 of 6 dims — high-frequency, needs resolution.

*Function class*
  ``gam``       binary; additive sum of univariate nonlinearities (sin / square / abs / tanh).
  ``mlp``       binary; random 2-layer tanh network — the smooth-nonlinear prior family.
  ``scm``       binary; small random causal DAG with neural mechanisms — correlated features
                that mix causes of the label with its effects, unlike every other recipe's
                i.i.d. design matrix.
  ``gp``        binary; smooth random function via random Fourier features (GP-like).
  ``tree``      binary; random axis-aligned depth-4 tree — piecewise constant, favours trees.

*Statistical regime*
  ``sparse``    binary; linear signal in 3 randomly-placed columns of 40 pure-noise ones.
  ``wide_sparse`` binary; 4 informative columns hidden among 60 distractors.
  ``wide_dense`` binary; linear signal spread across all 64 columns, with a stronger core.
  ``wide_nonlinear`` binary; nonlinear main effects and interactions hidden in 64 columns.
  ``imbalanced`` binary; 10:1 class prior — separates balanced accuracy from accuracy and
                exposes a model that falls back to the majority class.
  ``noisy``     binary; linear boundary at 25% label noise, i.e. a Bayes error of 25%.

*Input and output structure*
  ``mixed``     binary; categorical + numeric columns, label from a rule over both.
  ``hier``      6-class; two super-clusters each split into three fine-grained sub-classes.

``spec.fetch_id`` names the recipe, optionally with an explicit seed as ``"<recipe>:<seed>"``
(default seed 0). Feature columns are generically named ``x1..xd`` and categorical levels are
opaque tokens, so nothing here is nameable from memory. Generation uses only ``numpy`` so it
is reproducible independent of any library's RNG internals. Note: the generated table is
cached like any other dataset, so changing a recipe requires clearing the dataset cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabbench_llm.data.loaders.base import RawDataset

if TYPE_CHECKING:
    from tabbench_llm.data.datasets import DatasetSpec


def _feature_frame(matrix: np.ndarray) -> pd.DataFrame:
    """Wrap a feature matrix as a DataFrame with generic ``x1..xd`` column names."""
    return pd.DataFrame(matrix, columns=[f"x{i + 1}" for i in range(matrix.shape[1])])


def _median_split(scores: np.ndarray) -> np.ndarray:
    """Threshold a continuous score at its median -> balanced binary labels."""
    return (scores > np.median(scores)).astype(int)


def _flip(rng: np.random.Generator, y: np.ndarray, rate: float) -> np.ndarray:
    """Flip each binary label independently with probability *rate* (label noise)."""
    return np.where(rng.random(len(y)) < rate, 1 - y, y)


def _shuffle(rng: np.random.Generator, X: np.ndarray, y: np.ndarray):
    """Shuffle rows so their order does not encode the class."""
    order = rng.permutation(len(y))
    return X[order], y[order]


# ---------------------------------------------------------------------------
# Boundary geometry
# ---------------------------------------------------------------------------


def _linear(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task with a linear decision boundary over 8 features (~3% label noise)."""
    n, d = 400, 8
    X = rng.normal(size=(n, d))
    y = _flip(rng, _median_split(X @ rng.normal(size=d)), 0.03)
    return _feature_frame(X), y


def _rings(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task with a radial boundary in the first 2 of 6 dims (4 noise dims).

    Neither monotone in any feature nor axis-aligned, so it is the geometric opposite of
    ``tree``: a single split on either informative feature carries almost no information.
    """
    n, d, radius = 400, 6, 3.0
    X = rng.normal(size=(n, d))
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    # sqrt of a uniform gives points spread evenly over the disc rather than piled at the centre
    r = np.sqrt(rng.uniform(0.0, 1.0, n)) * radius
    X[:, 0], X[:, 1] = r * np.cos(angle), r * np.sin(angle)
    y = _flip(rng, (r > radius / np.sqrt(2.0)).astype(int), 0.02)  # equal-area split
    return _feature_frame(X), y


def _blobs(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """3-class task: Gaussian clusters separated in the first 2 of 6 dims (rest are noise)."""
    d, per = 6, 120
    centers = np.zeros((3, d))
    centers[0, :2] = (3.5, 0.0)
    centers[1, :2] = (-3.0, 3.0)
    centers[2, :2] = (-3.0, -3.0)
    X = np.vstack([rng.normal(centers[c], 1.0, size=(per, d)) for c in range(3)])
    X, y = _shuffle(rng, X, np.repeat([0, 1, 2], per))
    return _feature_frame(X), y


# ---------------------------------------------------------------------------
# Interaction order
# ---------------------------------------------------------------------------


def _xor(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task whose label is the XOR of two features amid 2 distractors (~2% noise)."""
    n = 320
    x1 = rng.uniform(-2.0, 2.0, n)
    x2 = rng.uniform(-2.0, 2.0, n)
    x3 = rng.normal(size=n)  # distractor
    x4 = rng.normal(size=n)  # distractor
    y = _flip(rng, ((x1 > 0) ^ (x2 > 0)).astype(int), 0.02)
    return _feature_frame(np.column_stack([x1, x2, x3, x4])), y


def _parity3(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary 3-way parity over 3 of 5 features (~2% noise).

    Every subset of two informative features is uninformative about the label, so the task
    is only solvable by recovering all three at once — a strictly harder interaction than
    ``xor`` and the sharpest test of in-context search in the set. Only two distractors:
    with more, RandomForest on the full table is already near chance, and a task no model
    can learn at any training size cannot discriminate between models.
    """
    n, d = 600, 5
    X = rng.uniform(-2.0, 2.0, (n, d))
    y = _flip(rng, ((X[:, 0] > 0) ^ (X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(int), 0.02)
    return _feature_frame(X), y


def _checkers(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary 3x3 checkerboard in the first 2 of 4 dims (2 noise dims, ~2% noise).

    Axis-aligned like ``tree`` but high-frequency: recovering it needs several splits per
    informative feature, so it separates resolution from hypothesis class. The grid is kept
    at 3x3 so the structure is recoverable from a few hundred rows.
    """
    n, d = 1000, 4
    X = rng.uniform(-1.5, 1.5, (n, d))
    cells = np.floor(X[:, 0]).astype(int) + np.floor(X[:, 1]).astype(int)
    y = _flip(rng, (cells % 2).astype(int), 0.02)
    return _feature_frame(X), y


# ---------------------------------------------------------------------------
# Function class
# ---------------------------------------------------------------------------


def _gam(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary additive task: a sum of univariate nonlinearities over 4 of 8 features."""
    n, d = 420, 8
    X = rng.uniform(-3.0, 3.0, (n, d))
    scores = np.sin(1.5 * X[:, 0]) + 0.4 * X[:, 1] ** 2 - np.abs(X[:, 2]) + np.tanh(2.0 * X[:, 3])
    return _feature_frame(X), _flip(rng, _median_split(scores), 0.02)


def _mlp(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task from a random 2-layer tanh network over 10 features (~2% noise).

    The smooth-nonlinear prior family that tabular foundation models are meta-trained on,
    so it is the setting most favourable to TabPFN and the natural reference point for the
    other function classes.
    """
    n, d, hidden = 500, 10, 16
    X = rng.normal(size=(n, d))
    w1 = rng.normal(size=(d, hidden)) / np.sqrt(d)
    b1 = rng.normal(size=hidden) * 0.5
    w2 = rng.normal(size=hidden) / np.sqrt(hidden)
    scores = np.tanh(X @ w1 + b1) @ w2
    return _feature_frame(X), _flip(rng, _median_split(scores), 0.02)


def _scm(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task generated by a small structural causal model with neural mechanisms.

    A random DAG over 12 latent nodes is sampled; each node is a small random nonlinear
    (tanh) function of up to 3 parents plus independent noise, and one node becomes the
    label. This is the prior family tabular foundation models are meta-trained on, and it
    differs from every other recipe here in one respect that matters: the features are not
    an i.i.d. design matrix. They are nodes of the same graph, so they are correlated with
    each other, and the observed set deliberately mixes **causes** of the label with its
    **effects** — the anticausal direction that dominates real tabular data and that a model
    reasoning about "which column explains the target" has to handle.
    """
    n, n_nodes, n_observed, max_parents = 500, 12, 8, 3
    nodes = np.zeros((n, n_nodes))
    parents: list[np.ndarray] = []

    for j in range(n_nodes):
        # Parents are drawn from earlier nodes only, which makes the graph acyclic by
        # construction (the index order is a topological order).
        n_par = min(j, int(rng.integers(0, max_parents + 1)))
        par = rng.choice(j, size=n_par, replace=False) if n_par else np.array([], dtype=int)
        parents.append(par)
        if len(par) == 0:
            nodes[:, j] = rng.normal(size=n)  # exogenous root
        else:
            hidden = 4
            w1 = rng.normal(size=(len(par), hidden))
            w2 = rng.normal(size=hidden)
            mechanism = np.tanh(nodes[:, par] @ w1 + rng.normal(size=hidden)) @ w2
            nodes[:, j] = mechanism + rng.normal(size=n) * 0.3
        nodes[:, j] /= nodes[:, j].std() + 1e-9  # keep scales comparable down the graph

    # The label must have both parents (causes to find) and children (effects to exploit),
    # otherwise the anticausal half of the point is lost.
    children = {j: [c for c in range(n_nodes) if j in parents[c]] for j in range(n_nodes)}
    eligible = [j for j in range(n_nodes) if len(parents[j]) > 0 and children[j]]
    assert eligible, "degenerate SCM: no node has both a parent and a child."
    label_node = int(rng.choice(eligible))

    # Observe a random subset of the remaining nodes, forcing in one cause and one effect.
    must = [int(rng.choice(parents[label_node])), int(rng.choice(children[label_node]))]
    rest = [j for j in range(n_nodes) if j != label_node and j not in must]
    chosen = must + list(rng.choice(rest, size=min(n_observed - 2, len(rest)), replace=False))
    return _feature_frame(nodes[:, sorted(chosen)]), _flip(
        rng, _median_split(nodes[:, label_node]), 0.02
    )


def _gp(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task from a smooth random function over 8 features (~2% noise).

    Built from random Fourier features, i.e. a draw from (an approximation of) a Gaussian
    process with an RBF kernel — smooth everywhere but with no parametric form to recover,
    unlike ``linear`` / ``gam`` / ``mlp``. The length-scale is kept long enough that the
    function is learnable from a few hundred rows rather than looking like noise.
    """
    n, d, n_features = 500, 6, 64
    X = rng.normal(size=(n, d))
    w = rng.normal(size=(d, n_features)) * 0.35  # inverse length-scale
    phase = rng.uniform(0.0, 2.0 * np.pi, n_features)
    basis = np.cos(X @ w + phase) * np.sqrt(2.0 / n_features)
    return _feature_frame(X), _flip(rng, _median_split(basis @ rng.normal(size=n_features)), 0.02)


def _tree(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task from a random axis-aligned depth-4 decision tree over 8 features.

    Piecewise constant on axis-aligned boxes — the hypothesis class RandomForest is exactly
    matched to, and the counterpart to ``rings`` (same difficulty, opposite geometry).
    """
    n, d, depth, min_leaf = 480, 8, 4, 12
    X = rng.normal(size=(n, d))

    leaves: list[np.ndarray] = []

    def grow(idx: np.ndarray, level: int) -> None:
        if level == depth or len(idx) < 2 * min_leaf:
            leaves.append(idx)
            return
        feature = int(rng.integers(0, d))
        threshold = float(rng.normal() * 0.6)
        left = X[idx, feature] <= threshold
        if left.sum() < min_leaf or (~left).sum() < min_leaf:
            leaves.append(idx)
            return
        grow(idx[left], level + 1)
        grow(idx[~left], level + 1)

    grow(np.arange(n), 0)

    # Label the leaves so the two classes end up near-balanced whatever shape the tree took:
    # walk the leaves largest-first and give each to whichever class is currently smaller.
    y = np.zeros(n, dtype=int)
    sizes = [0, 0]
    for leaf in sorted(leaves, key=len, reverse=True):
        cls = int(sizes[1] < sizes[0])
        y[leaf] = cls
        sizes[cls] += len(leaf)
    return _feature_frame(X), _flip(rng, y, 0.02)


# ---------------------------------------------------------------------------
# Statistical regime
# ---------------------------------------------------------------------------


def _sparse(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary linear task carried by 3 randomly-placed columns out of 40 pure-noise ones.

    The informative columns are drawn at random rather than placed first, so their position
    cannot be guessed — with only a few dozen few-shot rows this is a feature-selection
    problem before it is a classification problem.
    """
    n, d, n_informative = 400, 40, 3
    X = rng.normal(size=(n, d))
    weights = np.zeros(d)
    weights[rng.choice(d, size=n_informative, replace=False)] = rng.normal(size=n_informative) * 2.0
    return _feature_frame(X), _flip(rng, _median_split(X @ weights), 0.02)


def _wide_sparse(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """64-feature selection stress test with only 4 informative columns.

    The signal locations are redrawn from the recipe seed, so neither their positions nor the
    generated rows can be memorised. This extends ``sparse`` while remaining inside the common
    32K-token window at the headline's 100 training rows.
    """
    n, d, n_informative = 640, 64, 4
    X = rng.normal(size=(n, d))
    weights = np.zeros(d)
    informative = rng.choice(d, size=n_informative, replace=False)
    weights[informative] = rng.normal(size=n_informative) * 2.5
    return _feature_frame(X), _flip(rng, _median_split(X @ weights), 0.02)


def _wide_dense(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """64-feature linear task with signal distributed across every column.

    Twelve randomly located columns carry a stronger core and the other 52 carry weak signal.
    This keeps the task recoverable while distinguishing evidence aggregation from sparse
    feature discovery; no column is a pure distractor.
    """
    n, d, n_core = 800, 64, 12
    X = rng.normal(size=(n, d))
    weights = rng.normal(scale=0.15, size=d)
    core = rng.choice(d, size=n_core, replace=False)
    weights[core] += rng.normal(scale=1.0, size=n_core)
    return _feature_frame(X), _flip(rng, _median_split(X @ weights), 0.02)


def _wide_nonlinear(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """64-feature nonlinear task with 8 informative columns at random positions.

    Main effects (sine, square, absolute value, threshold) and pairwise interactions are mixed,
    while the remaining 56 columns are distractors. It probes whether degradation on a wide
    prompt is specific to feature search or persists after a nonlinear rule must be composed.
    """
    n, d, n_informative = 800, 64, 8
    X = rng.normal(size=(n, d))
    informative = rng.choice(d, size=n_informative, replace=False)
    z = X[:, informative]
    scores = (
        np.sin(1.4 * z[:, 0])
        + 0.45 * z[:, 1] ** 2
        - 0.8 * np.abs(z[:, 2])
        + 0.9 * (z[:, 3] * z[:, 4])
        + 0.8 * np.logical_xor(z[:, 5] > 0, z[:, 6] > 0)
        + 0.35 * z[:, 7]
    )
    return _feature_frame(X), _flip(rng, _median_split(scores), 0.02)


def _imbalanced(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary linear task with a 10:1 class prior over 6 features (~2% noise).

    Accuracy is 90% for a constant predictor here while balanced accuracy is 50%, so this is
    the recipe on which a model that quietly falls back to the majority class is visible.
    """
    n, d = 550, 6
    X = rng.normal(size=(n, d))
    scores = X @ rng.normal(size=d)
    y = _flip(rng, (scores > np.quantile(scores, 0.9)).astype(int), 0.02)
    return _feature_frame(X), y


def _noisy(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary linear task at 25% label noise over 6 features — a Bayes error of 25%.

    Caps the attainable accuracy at 0.75, so a model reporting more than that on the test
    split is reading noise, and confident probabilities are miscalibrated by construction.
    """
    n, d = 480, 6
    X = rng.normal(size=(n, d))
    return _feature_frame(X), _flip(rng, _median_split(X @ rng.normal(size=d)), 0.25)


# ---------------------------------------------------------------------------
# Input and output structure
# ---------------------------------------------------------------------------


def _mixed(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Binary task over mixed categorical + numeric columns, label from a rule spanning both.

    The only recipe that exercises the categorical path end to end (the LLM sees ``x1=c2``,
    AutoGluon encodes the level natively). Levels are opaque tokens, so nothing about them is
    nameable from memory.
    """
    n = 440
    cat_a = rng.choice([f"c{i}" for i in range(4)], n)
    cat_b = rng.choice([f"g{i}" for i in range(3)], n)
    num_a = rng.normal(size=n)
    num_b = rng.uniform(0.0, 10.0, n)
    num_c = rng.normal(size=n)  # distractor
    rule = (np.isin(cat_a, ["c0", "c2"]) & (num_b > 5.0)) | ((cat_b == "g1") & (num_a > 0.0))
    frame = pd.DataFrame({"x1": cat_a, "x2": cat_b, "x3": num_a, "x4": num_b, "x5": num_c}).astype(
        {"x1": "category", "x2": "category"}
    )
    return frame, _flip(rng, rule.astype(int), 0.02)


def _hier(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """6-class task: two well-separated super-clusters, each split into three close sub-classes.

    Coarse structure is easy and fine structure is hard, so the confusion matrix — not just
    the scalar score — says whether a model resolved the hierarchy or only its top level.
    """
    d, per = 8, 80
    supers = np.array([[5.0, 0.0], [-5.0, 0.0]])
    subs = np.array([[0.0, 1.2], [1.0, -0.6], [-1.0, -0.6]])
    blocks, labels = [], []
    for s in range(2):
        for k in range(3):
            center = np.zeros(d)
            center[:2] = supers[s]
            center[2:4] = subs[k]
            blocks.append(rng.normal(center, 0.6, size=(per, d)))
            labels.append(np.full(per, s * 3 + k))
    X, y = _shuffle(rng, np.vstack(blocks), np.concatenate(labels))
    return _feature_frame(X), y


#: recipe name -> generator. The registry's ``fetch_id`` selects one.
_RECIPES = {
    "linear": _linear,
    "rings": _rings,
    "blobs": _blobs,
    "xor": _xor,
    "parity3": _parity3,
    "checkers": _checkers,
    "gam": _gam,
    "mlp": _mlp,
    "scm": _scm,
    "gp": _gp,
    "tree": _tree,
    "sparse": _sparse,
    "wide_sparse": _wide_sparse,
    "wide_dense": _wide_dense,
    "wide_nonlinear": _wide_nonlinear,
    "imbalanced": _imbalanced,
    "noisy": _noisy,
    "mixed": _mixed,
    "hier": _hier,
}


class SyntheticLoader:
    """Generate a deterministic synthetic classification table from a named recipe."""

    def fetch(self, spec: DatasetSpec) -> RawDataset:
        """Generate the recipe named by ``spec.fetch_id`` and return it as a RawDataset."""
        recipe, _, seed_str = spec.fetch_id.partition(":")
        assert recipe in _RECIPES, (
            f"{spec.dataset_id}: unknown synthetic recipe {recipe!r} "
            f"(expected one of {sorted(_RECIPES)})."
        )
        seed = int(seed_str) if seed_str else 0
        X, y_arr = _RECIPES[recipe](np.random.default_rng(seed))
        y = pd.Series(y_arr, name="target")
        problem_type = spec.problem_type or ("multiclass" if y.nunique() > 2 else "binary")

        return RawDataset(
            dataset_id=spec.dataset_id,
            X=X,
            y=y,
            problem_type=problem_type,
            license="synthetic (generated; public domain)",
            source_url=f"synthetic:{recipe}:{seed}",
            citation=f"Synthetic dataset {spec.dataset_id} (recipe={recipe}, seed={seed}).",
            metadata={
                "recipe": recipe,
                "seed": seed,
                "n_features": int(X.shape[1]),
                "target": "target",
            },
        )
