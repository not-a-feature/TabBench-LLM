"""Seed management for multi-repetition benchmark runs."""


def get_seeds(config):
    """Return the list of split indices for the run.

    The returned integers are the per-split ``random_state`` values the pipeline iterates
    over — each one both seeds the split and names its ``seed_<n>/`` output directory.

    * ``cv_folds`` set to ``k`` — (repeated) k-fold CV: yields
      ``[0, 1, ..., k * repeats - 1]``, where ``repeats = n_repetitions or 1``. The
      benchmark reads each index ``g`` as ``repeat = g // k``, ``fold = g % k`` (see
      :meth:`tabbench_llm.benchmark.TabBenchLLM._kfold_split`).
    * ``cv_folds`` null, ``n_repetitions`` set — repeated random holdout: yields
      ``[0, 1, ..., n_repetitions-1]``.
    * both null — the single ``random_state`` seed.

    All three keys are required in the config (see
    :data:`tabbench_llm.config.REQUIRED_KEYS`) — none defaults.

    Parameters
    ----------
    config : dict
        Benchmark configuration dictionary.

    Returns
    -------
    list[int]
        Ordered list of split indices to iterate over.
    """
    n_repetitions = config["n_repetitions"]
    cv_folds = config["cv_folds"]
    if cv_folds is not None:
        repeats = n_repetitions if n_repetitions is not None else 1
        return list(range(cv_folds * repeats))
    if n_repetitions is not None:
        return list(range(n_repetitions))
    return [config["random_state"]]
