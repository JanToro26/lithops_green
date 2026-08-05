#!/usr/bin/env python3
"""
App 2/4 (Inigo Arriazu): Titanic (analitica de datos + entrenamiento).

Mix of CPU-bound (training) and memory-bound tasks. Each worker trains a Random Forest
(n_jobs=1, to avoid mixing multi-threading with multi-process parallelism) on
its own synthetic data chunk of fixed size -> strong scaling scenario
(more workers = more total data processed). Returns the accuracy (for monitoring)
and the number of samples.

Requires: numpy, scikit-learn.

    python examples/energy_apps/app_titanic.py
"""
from energy_report import profile_map

SAMPLES_PER_WORKER = 80_000   # fixed size per worker (strong scaling)


def titanic_worker(chunk_id):
    """Trains a Random Forest on a synthetic chunk (CPU-bound)."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(chunk_id)
    n = SAMPLES_PER_WORKER
    X = rng.normal(size=(n, 8))
    # label with signal + noise, for the model to have something to learn
    logit = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.5, size=n)
    y = (logit > 0).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=chunk_id)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=chunk_id)
    clf.fit(Xtr, ytr)
    return {'chunk': chunk_id, 'n_samples': n, 'accuracy': float(clf.score(Xte, yte))}


def make_iterdata(workers):
    return list(range(workers))


def summary(results):
    accs = [r['accuracy'] for r in results]
    total = sum(r['n_samples'] for r in results)
    return f"{total:,} muestras totales, accuracy media {sum(accs)/len(accs):.3f}"


if __name__ == '__main__':
    config_space = [
        {'workers': w, 'memory': m}
        for m in (1024, 2048)
        for w in (1, 2, 4, 8)
    ]
    profile_map('titanic_rf', titanic_worker, make_iterdata, config_space,
                reduce_fn=summary)