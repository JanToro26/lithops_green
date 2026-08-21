#!/usr/bin/env python3
"""
Predictive energy model (Task 3 scaffold).

Trains a small per-app model from the profiling CSVs, so it can PREDICT the
energy and time of resource configurations that were NOT measured, and then
optimize over the predicted grid (min-energy, or greenest under a time budget)
- including configurations you never ran.

Current data note: locally only the 'workers' axis varies energy (memory is not
enforced on localhost), so the model is workers-only for now. When you profile on
a real backend (AWS Lambda / K8s), memory/CPU also vary and become extra features
- the pipeline already accounts for that (it uses whichever inputs actually vary).

With few points per app, treat accuracy here as INDICATIVE, not a validation.

    python examples/energy_apps/model.py
    python examples/energy_apps/model.py --degree 2 --time-budget 8 --wmax 16
"""
import os
import argparse
import numpy as np

from optimize import load_points, greenest_under, pareto_front


def _fit(x, y, degree):
    """Fit y ~ poly(x); cap degree so we never over-fit beyond the data."""
    deg = max(1, min(degree, len(set(x)) - 1))
    return np.poly1d(np.polyfit(x, y, deg)), deg


def train_app(points, degree):
    """Fit energy(workers) and time(workers) for one app."""
    w = np.array([p['workers'] for p in points], float)
    e = np.array([p['energy_j'] for p in points], float)
    t = np.array([p['time_s'] for p in points], float)
    e_model, e_deg = _fit(w, e, degree)
    t_model, t_deg = _fit(w, t, degree)
    return {'energy': e_model, 'time': t_model, 'e_deg': e_deg, 't_deg': t_deg}


def predict_grid(model, wmin, wmax, memory=''):
    """Predict energy/time for every worker count in [wmin, wmax]."""
    grid = []
    for w in range(wmin, wmax + 1):
        grid.append({
            'workers': w, 'memory': memory,
            'energy_j': float(model['energy'](w)),
            'time_s': float(model['time'](w)),
            'source': 'predicted',
        })
    return grid


def loo_error(points, degree):
    """Leave-one-out % error for the energy model (indicative with small N)."""
    if len(points) < 3:
        return None
    errs = []
    for i in range(len(points)):
        train = [p for j, p in enumerate(points) if j != i]
        test = points[i]
        w = np.array([p['workers'] for p in train], float)
        e = np.array([p['energy_j'] for p in train], float)
        m, _ = _fit(w, e, degree)
        pred = float(m(test['workers']))
        if test['energy_j']:
            errs.append(abs(pred - test['energy_j']) / test['energy_j'] * 100)
    return sum(errs) / len(errs) if errs else None


def plot(app, points, grid, out_dir, time_budget=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not available; skipping plot)")
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter([p['workers'] for p in points], [p['energy_j'] for p in points],
                color='steelblue', zorder=5, label='measured')
    plt.plot([g['workers'] for g in grid], [g['energy_j'] for g in grid],
             color='crimson', label='predicted energy')
    if time_budget is not None:
        best = greenest_under(grid, time_budget)
        if best:
            plt.scatter([best['workers']], [best['energy_j']], s=220, marker='*',
                        color='gold', edgecolors='black', zorder=6,
                        label=f'chosen (t<={time_budget}s)')
    plt.xlabel('Workers')
    plt.ylabel('Energy per work unit (J)')
    plt.title(f'Energy model - {app}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, f'model_{app}.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  plot saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='*', default=None)
    ap.add_argument('--degree', type=int, default=1, help='polynomial degree (default 1)')
    ap.add_argument('--wmax', type=int, default=16, help='max workers to predict up to')
    ap.add_argument('--time-budget', type=float, default=None)
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    results = os.path.join(here, 'results')
    import glob
    csv_paths = args.csv or glob.glob(os.path.join(results, 'profiling*_avg.csv'))
    apps = load_points(csv_paths)
    if not apps:
        print("No profiling data found.")
        return

    for app, points in apps.items():
        memory = points[0]['memory']
        model = train_app(points, args.degree)
        grid = predict_grid(model, 1, args.wmax, memory=memory)
        err = loo_error(points, args.degree)

        print(f"\n=== {app} ({points[0]['source']} energy) ===")
        print(f"  fitted energy(workers): degree {model['e_deg']}"
              + (f"  | leave-one-out error ~{err:.1f}% (indicative, N={len(points)})"
                 if err is not None else "  | too few points for LOO"))
        # Predicted optimum
        min_e = min(grid, key=lambda g: g['energy_j'])
        print(f"  predicted lowest energy (1..{args.wmax}w): "
              f"workers={min_e['workers']} E={min_e['energy_j']:.1f}J t={min_e['time_s']:.2f}s")
        if args.time_budget is not None:
            best = greenest_under(grid, args.time_budget)
            if best:
                print(f"  predicted greenest under {args.time_budget}s: "
                      f"workers={best['workers']} E={best['energy_j']:.1f}J t={best['time_s']:.2f}s")
            else:
                print(f"  predicted greenest under {args.time_budget}s: (none)")

        if not args.no_plot:
            plot(app, points, grid, os.path.join(results, 'images'),
                 time_budget=args.time_budget)


if __name__ == '__main__':
    main()