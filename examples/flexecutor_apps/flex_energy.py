#!/usr/bin/env python3
"""
Energy reader for Flexecutor runs.

Flexecutor executes each Stage through Lithops (see flexecutor's ThreadPoolProcessor,
which calls FunctionExecutor.map). Therefore every stage invocation passes through
the Lithops worker handler and the EnergyManager, and the resulting per-worker
stats are exposed by StageFuture.stats.

This module aggregates those stats (summing energy across all workers of all
stages, taking the slowest stage duration) and appends one row per run to a CSV
whose schema matches examples/energy_apps so the same optimizer can consume both.
"""
import os
import csv
import time
import statistics

K_RAPL = 'worker_func_rapl_energy_pkg'
K_PSU = 'worker_func_psutil_energy_pkg'
K_DUR = 'worker_func_energy_duration'
K_CPU = 'worker_func_psutil_avg_cpu_percent'
K_METHOD = 'worker_func_energy_method_used'

_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DEFAULT_RAW = os.path.join(_RESULTS, 'profiling_flex_raw.csv')
DEFAULT_AVG = os.path.join(_RESULTS, 'profiling_flex_avg.csv')

RAW_HEADER = ['app', 'workers', 'memory', 'repeat', 'wall_s', 'max_duration_s',
              'psutil_pkg_j', 'rapl_pkg_j', 'avg_cpu_pct', 'method', 'tstamp']


def _stats_of(stage_future):
    """Return the list of per-worker Lithops stat dicts for a StageFuture."""
    s = stage_future.stats            # StageFuture.stats -> [f.stats for f in future]
    return [d for d in s if isinstance(d, dict)]


def summarize_flex(futures_dict):
    """
    Aggregate energy across ALL stages of a Flexecutor DAG run.

    futures_dict: {stage_id: StageFuture} as returned by DAGExecutor.execute().
    """
    all_stats = []
    for sf in futures_dict.values():
        all_stats.extend(_stats_of(sf))
    n = len(all_stats)
    return {
        'psutil_pkg_j': sum(d.get(K_PSU, 0.0) for d in all_stats),
        'rapl_pkg_j': sum(d.get(K_RAPL, 0.0) for d in all_stats),
        'max_duration_s': max((d.get(K_DUR, 0.0) for d in all_stats), default=0.0),
        'avg_cpu_pct': (sum(d.get(K_CPU, 0.0) for d in all_stats) / n) if n else 0.0,
        'method': all_stats[0].get(K_METHOD, 'n/a') if all_stats else 'n/a',
        'n_invocations': n,
    }


def record_run(app, workers, memory, repeat, futures_dict, wall, csv_path=None):
    """Aggregate a Flexecutor run and append one RAW row; rebuild the AVG file."""
    raw = csv_path or DEFAULT_RAW
    summ = summarize_flex(futures_dict)
    os.makedirs(os.path.dirname(os.path.abspath(raw)), exist_ok=True)
    new = not os.path.exists(raw)
    with open(raw, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(RAW_HEADER)
        w.writerow([app, workers, memory, repeat, round(wall, 3),
                    round(summ['max_duration_s'], 3), round(summ['psutil_pkg_j'], 3),
                    round(summ['rapl_pkg_j'], 3), round(summ['avg_cpu_pct'], 2),
                    summ['method'], time.strftime('%Y-%m-%d %H:%M:%S')])
    _write_averaged(raw, DEFAULT_AVG)
    print(f"[{app}] workers={workers} rep={repeat} "
          f"dur={summ['max_duration_s']:.2f}s E_psutil={summ['psutil_pkg_j']:.2f}J "
          f"E_rapl={summ['rapl_pkg_j']:.2f}J invocations={summ['n_invocations']} "
          f"[{summ['method']}]")
    return summ


def _write_averaged(raw_path, avg_path):
    if not os.path.exists(raw_path):
        return
    groups = {}
    with open(raw_path, newline='') as f:
        for row in csv.DictReader(f):
            groups.setdefault((row['app'], row['workers'], row['memory']), []).append(row)
    rows = []
    for (app, workers, memory), items in groups.items():
        psu = [float(r['psutil_pkg_j']) for r in items]
        dur = [float(r['max_duration_s']) for r in items]
        mean = sum(psu) / len(psu) if psu else 0.0
        std = statistics.stdev(psu) if len(psu) > 1 else 0.0
        rows.append({
            'app': app, 'workers': workers, 'memory': memory, 'n': len(items),
            'psutil_pkg_j_mean': round(mean, 3), 'psutil_pkg_j_std': round(std, 3),
            'psutil_pkg_j_cv_pct': round(100 * std / mean, 2) if mean else 0.0,
            'rapl_pkg_j_mean': round(sum(float(r['rapl_pkg_j']) for r in items) / len(items), 3),
            'max_duration_s_mean': round(sum(dur) / len(dur), 3),
            'method': items[0]['method'],
        })
    rows.sort(key=lambda r: (r['app'], r['memory'], r['workers']))
    os.makedirs(os.path.dirname(os.path.abspath(avg_path)), exist_ok=True)
    with open(avg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['app'])
        if rows:
            w.writeheader()
            w.writerows(rows)