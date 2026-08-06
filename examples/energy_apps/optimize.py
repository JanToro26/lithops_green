#!/usr/bin/env python3
"""
Naive energy optimizer over the profiling data.

Reads the averaged profiling CSV(s) produced by the apps (energy_apps and
flexecutor_apps), and for each application:

  - builds the (energy, time) points across resource configurations,
  - computes the Pareto front (configs not dominated in BOTH energy and time),
  - recommends configurations:
        * min-energy            (lowest energy overall),
        * min-time              (fastest),
        * min-energy under a time budget  (--time-budget T): the greenest config
          that still finishes within T seconds  (Jolteon-style bounded execution),
  - optionally plots energy vs. time per app (matplotlib), highlighting the front,
    the time budget (if given) and the chosen config.

Energy source: uses RAPL if available (rapl_pkg_j_mean > 0), otherwise the psutil
estimate. Time: max_duration_s_mean (the slowest worker of the run).

    python examples/energy_apps/optimize.py
    python examples/energy_apps/optimize.py --time-budget 8
    python examples/energy_apps/optimize.py --no-plot
"""
import os
import csv
import glob
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, 'results')
IMAGES = os.path.join(RESULTS, 'images')


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_points(csv_paths):
    """Return {app: [ {workers, memory, energy_j, time_s, source}, ... ]}."""
    apps = {}
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                app = row.get('app')
                if not app:
                    continue
                rapl = _num(row.get('rapl_pkg_j_mean'), 0.0) or 0.0
                psu = _num(row.get('psutil_pkg_j_mean'), 0.0) or 0.0
                energy = rapl if rapl > 0 else psu
                time_s = _num(row.get('max_duration_s_mean'), 0.0) or 0.0
                apps.setdefault(app, []).append({
                    'workers': int(_num(row.get('workers'), 0) or 0),
                    'memory': row.get('memory') or '',
                    'energy_j': energy,
                    'time_s': time_s,
                    'source': 'rapl' if rapl > 0 else 'psutil',
                })
    return apps


def pareto_front(points):
    """Non-dominated points minimizing BOTH energy and time."""
    front = []
    for p in points:
        dominated = any(
            q is not p
            and q['energy_j'] <= p['energy_j'] and q['time_s'] <= p['time_s']
            and (q['energy_j'] < p['energy_j'] or q['time_s'] < p['time_s'])
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p['time_s'])


def greenest_under(points, budget):
    """Lowest-energy config whose time <= budget, or None if none qualifies."""
    ok = [p for p in points if p['time_s'] <= budget]
    return min(ok, key=lambda p: p['energy_j']) if ok else None


def _fmt(p):
    return f"workers={p['workers']:<2} mem={p['memory']:<6} E={p['energy_j']:8.1f}J t={p['time_s']:6.2f}s"


def report(app, points, time_budget=None):
    front = pareto_front(points)
    front_set = set(id(p) for p in front)
    min_e = min(points, key=lambda p: p['energy_j'])
    min_t = min(points, key=lambda p: p['time_s'])

    print(f"\n=== {app} ({points[0]['source']} energy) ===")
    for p in sorted(points, key=lambda p: (p['workers'])):
        tags = []
        if id(p) in front_set:
            tags.append('PARETO')
        if p is min_e:
            tags.append('min-energy')
        if p is min_t:
            tags.append('min-time')
        print(f"  {_fmt(p)}   {' '.join(tags)}")

    print(f"  -> lowest energy : {_fmt(min_e)}")
    print(f"  -> fastest       : {_fmt(min_t)}")
    if time_budget is not None:
        best = greenest_under(points, time_budget)
        if best:
            print(f"  -> greenest under {time_budget}s : {_fmt(best)}")
        else:
            print(f"  -> greenest under {time_budget}s : (no config meets the budget)")
    return front


def plot(app, points, front, out_dir, time_budget=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not available; skipping plot)")
        return
    os.makedirs(out_dir, exist_ok=True)
    xs = [p['time_s'] for p in points]
    ys = [p['energy_j'] for p in points]
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, color='steelblue', label='configs')
    for p in points:
        plt.annotate(f"{p['workers']}w", (p['time_s'], p['energy_j']),
                     textcoords="offset points", xytext=(5, 4), fontsize=8)
    plt.plot([p['time_s'] for p in front], [p['energy_j'] for p in front],
             color='crimson', marker='o', label='Pareto front')

    # Time budget: draw the line and highlight the chosen config
    if time_budget is not None:
        plt.axvline(time_budget, color='gray', linestyle='--', alpha=0.7,
                    label=f'budget = {time_budget}s')
        best = greenest_under(points, time_budget)
        if best:
            plt.scatter([best['time_s']], [best['energy_j']], s=220, marker='*',
                        color='gold', edgecolors='black', zorder=5,
                        label='chosen (greenest under budget)')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title(f'Energy vs time - {app}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    suffix = f'_budget{int(time_budget)}' if time_budget is not None else ''
    out = os.path.join(out_dir, f'pareto_{app}{suffix}.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  plot saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='*', default=None,
                    help='profiling avg CSV(s); default: results/profiling*_avg.csv')
    ap.add_argument('--time-budget', type=float, default=None,
                    help='seconds; report/plot the greenest config finishing within this time')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    csv_paths = args.csv or glob.glob(os.path.join(RESULTS, 'profiling*_avg.csv')) \
        or [os.path.join(RESULTS, 'profiling_avg.csv'),
            os.path.join(RESULTS, 'profiling_flex_avg.csv')]

    apps = load_points(csv_paths)
    if not apps:
        print(f"No profiling data found in: {csv_paths}")
        return

    for app, points in apps.items():
        front = report(app, points, time_budget=args.time_budget)
        if not args.no_plot:
            plot(app, points, front, IMAGES, time_budget=args.time_budget)


if __name__ == '__main__':
    main()