#!/usr/bin/env python3
"""
Shared energy-profiling harness for the parallel apps.

Runs a Lithops app across a set of configurations (number of workers, and
optionally memory), reads the per-invocation energy from future.stats, aggregates
it per configuration, and writes TWO CSVs:

  - RAW  (profiling_raw.csv): one row per repeat -> every individual measurement,
    so you can see the spread/margin you are working with.
  - AVG  (profiling_avg.csv): one row per configuration -> mean and spread (std,
    and cv_pct = std/mean*100) across repeats. This is the clean input for the
    optimization step.

The AVG file is regenerated from the RAW file after each run, so it always
reflects all measurements collected so far.

Usage from an app:
    from energy_report import profile_map
    profile_map('my_app', my_function, make_iterdata, config_space, repeats=5)
"""
import os
import csv
import time
import statistics

# Keys written by EnergyManager / SystemMonitor into future.stats
K_RAPL = 'worker_func_rapl_energy_pkg'
K_PSU = 'worker_func_psutil_energy_pkg'
K_DUR = 'worker_func_energy_duration'
K_CPU = 'worker_func_psutil_avg_cpu_percent'
K_METHOD = 'worker_func_energy_method_used'

# Output locations resolved relative to THIS file (not the current working
# directory), so every run writes to the same predictable place.
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DEFAULT_RAW = os.path.join(_RESULTS, 'profiling_raw.csv')
DEFAULT_AVG = os.path.join(_RESULTS, 'profiling_avg.csv')

RAW_HEADER = ['app', 'workers', 'memory', 'repeat', 'wall_s', 'max_duration_s',
              'psutil_pkg_j', 'rapl_pkg_j', 'avg_cpu_pct', 'method', 'tstamp']


def summarize(futures):
    """Aggregate the energy of all workers of a single execution."""
    stats = [getattr(f, 'stats', {}) or {} for f in futures]
    n = len(stats)
    return {
        'workers': n,
        # Total energy = sum over all workers (each measures its own share)
        'rapl_pkg_j': sum(s.get(K_RAPL, 0.0) for s in stats),
        'psutil_pkg_j': sum(s.get(K_PSU, 0.0) for s in stats),
        # Time = the slowest worker (workers run in parallel)
        'max_duration_s': max((s.get(K_DUR, 0.0) for s in stats), default=0.0),
        'avg_cpu_pct': (sum(s.get(K_CPU, 0.0) for s in stats) / n) if n else 0.0,
        'method': stats[0].get(K_METHOD, 'n/a') if stats else 'n/a',
    }


def print_summary(app, cfg, summ, wall, repeat, repeats):
    print(f"[{app}] rep {repeat}/{repeats} workers={summ['workers']:<3} "
          f"mem={cfg.get('memory')} wall={wall:6.2f}s dur={summ['max_duration_s']:6.2f}s "
          f"E_psutil={summ['psutil_pkg_j']:8.2f}J E_rapl={summ['rapl_pkg_j']:8.2f}J "
          f"cpu={summ['avg_cpu_pct']:4.1f}% [{summ['method']}]")


def append_raw(path, app, cfg, summ, wall, repeat):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(RAW_HEADER)
        w.writerow([app, summ['workers'], cfg.get('memory'), repeat, round(wall, 3),
                    round(summ['max_duration_s'], 3), round(summ['psutil_pkg_j'], 3),
                    round(summ['rapl_pkg_j'], 3), round(summ['avg_cpu_pct'], 2),
                    summ['method'], time.strftime('%Y-%m-%d %H:%M:%S')])


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _std(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


def write_averaged(raw_path, avg_path):
    """(Re)build the averaged CSV from the raw CSV, grouping by (app, workers, memory)."""
    if not os.path.exists(raw_path):
        return
    groups = {}
    with open(raw_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['app'], row['workers'], row['memory'])
            groups.setdefault(key, []).append(row)

    rows = []
    for (app, workers, memory), items in groups.items():
        psu = [float(r['psutil_pkg_j']) for r in items]
        rapl = [float(r['rapl_pkg_j']) for r in items]
        dur = [float(r['max_duration_s']) for r in items]
        wall = [float(r['wall_s']) for r in items]
        cpu = [float(r['avg_cpu_pct']) for r in items]
        psu_mean, psu_std = _mean(psu), _std(psu)
        rows.append({
            'app': app, 'workers': workers, 'memory': memory, 'n': len(items),
            'psutil_pkg_j_mean': round(psu_mean, 3),
            'psutil_pkg_j_std': round(psu_std, 3),
            'psutil_pkg_j_cv_pct': round(100 * psu_std / psu_mean, 2) if psu_mean else 0.0,
            'rapl_pkg_j_mean': round(_mean(rapl), 3),
            'max_duration_s_mean': round(_mean(dur), 3),
            'max_duration_s_std': round(_std(dur), 3),
            'wall_s_mean': round(_mean(wall), 3),
            'avg_cpu_pct_mean': round(_mean(cpu), 2),
            'method': items[0]['method'],
        })

    # Sort for readability: app, then memory, then workers (numeric)
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return -1.0
    rows.sort(key=lambda r: (r['app'], _num(r['memory']), _num(r['workers'])))

    fields = ['app', 'workers', 'memory', 'n',
              'psutil_pkg_j_mean', 'psutil_pkg_j_std', 'psutil_pkg_j_cv_pct',
              'rapl_pkg_j_mean', 'max_duration_s_mean', 'max_duration_s_std',
              'wall_s_mean', 'avg_cpu_pct_mean', 'method']
    os.makedirs(os.path.dirname(os.path.abspath(avg_path)), exist_ok=True)
    with open(avg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def profile_map(app, func, make_iterdata, config_space,
                backend='localhost', storage='localhost',
                csv_path=None, avg_path=None, reduce_fn=None, repeats=1):
    """
    Run 'func' for each configuration in 'config_space' (dicts with at least
    {'workers': N} and optionally {'memory': MB}). Each config runs 'repeats'
    times; every repeat appends a RAW row and the AVG file is rebuilt at the end.
    """
    import lithops
    raw_path = csv_path or DEFAULT_RAW
    avg_path = avg_path or DEFAULT_AVG
    print(f"== Profiling '{app}' on backend '{backend}' (repeats={repeats}) ==")
    for cfg in config_space:
        workers = cfg['workers']
        memory = cfg.get('memory')
        iterdata = make_iterdata(workers)
        for r in range(1, repeats + 1):
            kwargs = dict(backend=backend, storage=storage)
            if memory:
                kwargs['runtime_memory'] = memory
            fexec = lithops.FunctionExecutor(**kwargs)
            t0 = time.time()
            futs = fexec.map(func, iterdata)
            results = fexec.get_result(fs=futs)
            wall = time.time() - t0
            summ = summarize(futs)
            print_summary(app, cfg, summ, wall, r, repeats)
            append_raw(raw_path, app, cfg, summ, wall, r)
            fexec.clean()
            if reduce_fn is not None:
                try:
                    print(f"    result: {reduce_fn(results)}")
                except Exception as e:
                    print(f"    (reduce_fn failed: {e})")
    write_averaged(raw_path, avg_path)
    print(f"Raw:      {raw_path}")
    print(f"Averaged: {avg_path}")


def profile_pipeline(app, run_fn, config_space,
                     backend='localhost', storage='localhost',
                     csv_path=None, avg_path=None, repeats=1):
    """
    Like profile_map but for MULTI-STAGE workflows.

    run_fn(fexec, workers) runs all stages and returns the list of ALL futures
    produced. Energy is aggregated over every invocation of every stage.
    """
    import lithops
    raw_path = csv_path or DEFAULT_RAW
    avg_path = avg_path or DEFAULT_AVG
    print(f"== Profiling '{app}' (pipeline) on backend '{backend}' (repeats={repeats}) ==")
    for cfg in config_space:
        workers = cfg['workers']
        memory = cfg.get('memory')
        for r in range(1, repeats + 1):
            kwargs = dict(backend=backend, storage=storage)
            if memory:
                kwargs['runtime_memory'] = memory
            fexec = lithops.FunctionExecutor(**kwargs)
            t0 = time.time()
            all_futs = run_fn(fexec, workers)
            wall = time.time() - t0
            summ = summarize(all_futs)
            summ['workers'] = workers   # configured workers (not total invocations)
            print_summary(app, cfg, summ, wall, r, repeats)
            append_raw(raw_path, app, cfg, summ, wall, r)
            fexec.clean()
    write_averaged(raw_path, avg_path)
    print(f"Raw:      {raw_path}")
    print(f"Averaged: {avg_path}")