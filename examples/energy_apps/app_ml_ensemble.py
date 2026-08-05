#!/usr/bin/env python3
"""
App 3/4 (Inigo Arriazu): ML Ensemble Pipeline (multi-stage workflow).

Four dependent stages:
  Stage0 - PCA: dimensionality reduction of synthetic data (1 worker).
  Stage1 - Train: each worker trains a distinct model of the ensemble (W workers)
           -> more workers = more models = more total work.
  Stage2 - Aggregate: combine the models' metrics (1 worker).
  Stage3 - Test: evaluate the ensemble on fresh data (1 worker).

Energy is aggregated over ALL invocations of ALL stages.
Requires: numpy, scikit-learn. (Inigo used LightGBM; here GradientBoosting from
sklearn to minimize dependencies; it can be swapped for lightgbm.)

    python examples/energy_apps/app_ml_ensemble.py
"""
from energy_report import profile_pipeline

N_SAMPLES = 20_000
N_FEATURES = 50
N_COMPONENTS = 10


def _make_data(seed, n=N_SAMPLES, d=N_FEATURES):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = ((X @ w + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    return X, y


def stage0_pca(seed):
    """PCA for dimensionality reduction (CPU)."""
    from sklearn.decomposition import PCA
    X, _ = _make_data(seed)
    pca = PCA(n_components=N_COMPONENTS, random_state=seed)
    pca.fit(X)
    return {'seed': seed, 'explained': float(pca.explained_variance_ratio_.sum())}


def stage1_train(model_id):
    """Train one ensemble model on PCA-reduced data (CPU-heavy)."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    X, y = _make_data(model_id)
    Xr = PCA(n_components=N_COMPONENTS, random_state=0).fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xr, y, test_size=0.2, random_state=model_id)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=model_id)
    clf.fit(Xtr, ytr)
    return {'model_id': model_id, 'accuracy': float(clf.score(Xte, yte))}


def stage2_aggregate(results):
    """Aggregate the ensemble models' metrics (lightweight)."""
    accs = [r['accuracy'] for r in results]
    return {'n_models': len(accs), 'mean_acc': sum(accs) / len(accs),
            'best_acc': max(accs)}


def stage3_test(agg):
    """Evaluate the ensemble on fresh data (moderate)."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = _make_data(999)
    Xr = PCA(n_components=N_COMPONENTS, random_state=0).fit_transform(X)
    clf = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(Xr, y)
    return {'final_score': float(clf.score(Xr, y)), 'ensemble': agg}


def run_pipeline(fexec, workers):
    """Run the 4 stages and return ALL futures produced."""
    f0 = fexec.map(stage0_pca, [42])
    fexec.get_result(fs=f0)

    f1 = fexec.map(stage1_train, list(range(workers)))
    r1 = fexec.get_result(fs=f1)

    f2 = fexec.map(stage2_aggregate, [r1])   # 1 task receiving the list of results
    r2 = fexec.get_result(fs=f2)

    f3 = fexec.map(stage3_test, [{'agg': r2[0]}])
    fexec.get_result(fs=f3)

    return list(f0) + list(f1) + list(f2) + list(f3)


if __name__ == '__main__':
    config_space = [
        {'workers': w, 'memory': m}
        for m in (1024, 2048)
        for w in (1, 2, 4, 8)
    ]
    profile_pipeline('ml_ensemble', run_pipeline, config_space)