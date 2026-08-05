#!/usr/bin/env python3
"""
Shared energy-profiling harness for the parallel apps.

Runs a Lithops app across a set of configurations (number of workers, and
optionally memory), reads the per-invocation energy from future.stats, aggregates
it per configuration, and appends it to a CSV. That CSV is the "profiling" that
the later optimization step consumes (energy vs. configuration).

Each configuration can be run several times (repeats) to average out the noise of
localhost measurements; every repeat writes its own CSV row (see the 'repeat'
column), so the raw measurements are preserved for later averaging.

Usage from an app:
    from energy_report import profile_map
    profile_map('my_app', my_function, make_iterdata, config_space, repeats=5)
"""
import os
import csv
import time

# Keys written by EnergyManager / SystemMonitor into future.stats
K_RAPL = 'worker_func_rapl_energy_pkg'
K_PSU = 'worker_func_psutil_energy_pkg'
K_DUR = 'worker_func_energy_duration'
K_CPU = 'worker_func_psutil_avg_cpu_percent'
K_METHOD = 'worker_func_energy_method_used'

# Default output location, resolved relative to THIS file (not the current
# working directory), so every run writes to the same predictable CSV.
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', 'profiling.csv')


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


def append_csv(path, app, cfg, summ, wall, repeat):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['app', 'workers', 'memory', 'repeat', 'wall_s', 'max_duration_s',
                        'psutil_pkg_j', 'rapl_pkg_j', 'avg_cpu_pct', 'method', 'tstamp'])
        w.writerow([app, summ['workers'], cfg.get('memory'), repeat, round(wall, 3),
                    round(summ['max_duration_s'], 3), round(summ['psutil_pkg_j'], 3),
                    round(summ['rapl_pkg_j'], 3), round(summ['avg_cpu_pct'], 2),
                    summ['method'], time.strftime('%Y-%m-%d %H:%M:%S')])


def _run_one_map(fexec_factory, func, iterdata, reduce_fn):
    fexec = fexec_factory()
    t0 = time.time()
    futs = fexec.map(func, iterdata)
    results = fexec.get_result(fs=futs)
    wall = time.time() - t0
    summ = summarize(futs)
    fexec.clean()
    extra = None
    if reduce_fn is not None:
        try:
            extra = reduce_fn(results)
        except Exception as e:
            extra = f"(reduce_fn failed: {e})"
    return summ, wall, extra


def profile_map(app, func, make_iterdata, config_space,
                backend='localhost', storage='localhost',
                csv_path=None, reduce_fn=None, repeats=1):
    """
    Run 'func' for each configuration in 'config_space' (dicts with at least
    {'workers': N} and optionally {'memory': MB}), collecting energy. Each config
    is executed 'repeats' times; every repeat writes its own CSV row.
    """
    csv_path = csv_path or DEFAULT_CSV
    print(f"== Profiling '{app}' on backend '{backend}' (repeats={repeats}) ==")
    for cfg in config_space:
        workers = cfg['workers']
        memory = cfg.get('memory')
        iterdata = make_iterdata(workers)

        def factory():
            import lithops
            kwargs = dict(backend=backend, storage=storage)
            if memory:
                kwargs['runtime_memory'] = memory
            return lithops.FunctionExecutor(**kwargs)

        for r in range(1, repeats + 1):
            summ, wall, extra = _run_one_map(factory, func, iterdata, reduce_fn)
            print_summary(app, cfg, summ, wall, r, repeats)
            append_csv(csv_path, app, cfg, summ, wall, r)
            if extra is not None:
                print(f"    result: {extra}")
    print(f"Profiling saved to: {csv_path}")


def profile_pipeline(app, run_fn, config_space,
                     backend='localhost', storage='localhost',
                     csv_path=None, repeats=1):
    """
    Like profile_map but for MULTI-STAGE workflows.

    run_fn(fexec, workers) runs all stages and returns the list of ALL futures
    produced. Energy is aggregated over every invocation of every stage. Each
    config is executed 'repeats' times; every repeat writes its own CSV row.
    """
    import lithops
    csv_path = csv_path or DEFAULT_CSV
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
            append_csv(csv_path, app, cfg, summ, wall, r)
            fexec.clean()
    print(f"Profiling saved to: {csv_path}")