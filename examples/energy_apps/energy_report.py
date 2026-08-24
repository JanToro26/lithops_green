#!/usr/bin/env python3
"""Energy-profiling harness for the parallel apps.

Runs an app across a set of configurations, reads per-invocation energy from
future.stats, and writes two CSVs: profiling_raw.csv (one row per repeat) and
profiling_avg.csv (one row per configuration, rebuilt from the raw file after
every run).

Multi-stage workflows get one row per stage plus a whole-pipeline row with
stage='ALL'.

    from energy_report import profile_map
    profile_map('my_app', my_function, make_iterdata, config_space, repeats=5)
"""
import os
import csv
import time
import statistics

# Keys written by EnergyManager / SystemMonitor into future.stats
K_RAPL = 'worker_func_rapl_energy_pkg'
K_PSU = 'worker_func_psutil_energy_pkg'                     # dynamic share, summable
K_PSU_IDLE_W = 'worker_func_psutil_p_idle_machine_w'        # machine floor, add once
K_DUR = 'worker_func_energy_duration'
K_CPU = 'worker_func_psutil_avg_cpu_percent'
K_METHOD = 'worker_func_energy_method_used'
K_TDP = 'worker_func_psutil_cpu_tdp_ref'
K_TDP_DEFAULT = 'worker_func_psutil_cpu_tdp_is_default'
K_CPU_MODEL = 'worker_func_psutil_cpu_model'
K_PERF = 'worker_func_perf_energy_pkg'

# Resolved relative to this file, not the working directory.
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DEFAULT_RAW = os.path.join(_RESULTS, 'profiling_raw.csv')
DEFAULT_AVG = os.path.join(_RESULTS, 'profiling_avg.csv')

RAW_HEADER = [
    'app', 'stage', 'workers', 'memory', 'work_units', 'repeat',
    'wall_s', 'max_duration_s',
    'psutil_dynamic_j', 'psutil_idle_j', 'psutil_pkg_j',
    'rapl_pkg_j', 'perf_pkg_j',
    'avg_cpu_pct', 'tdp_w', 'tdp_default', 'cpu_model', 'method', 'tstamp',
]

AVG_HEADER = [
    'app', 'stage', 'workers', 'memory', 'work_units', 'n',
    'psutil_pkg_j_mean', 'psutil_pkg_j_std', 'psutil_pkg_j_cv_pct',
    'psutil_dynamic_j_mean', 'psutil_idle_j_mean',
    'rapl_pkg_j_mean', 'rapl_pkg_j_std', 'perf_pkg_j_mean',
    'psutil_j_per_unit', 'rapl_j_per_unit', 'psutil_vs_rapl_pct',
    'max_duration_s_mean', 'max_duration_s_std',
    'wall_s_mean', 'avg_cpu_pct_mean',
    'tdp_w', 'tdp_default', 'cpu_model', 'method',
]

PIPELINE_STAGE = 'ALL'


def _stats_of(futures):
    return [getattr(f, 'stats', {}) or {} for f in futures]


def summarize(futures, stage=PIPELINE_STAGE):
    """Aggregate the energy of all workers of one stage."""
    stats = _stats_of(futures)
    n = len(stats)
    if not n:
        return {'stage': stage, 'workers': 0, 'psutil_dynamic_j': 0.0,
                'psutil_idle_j': 0.0, 'psutil_pkg_j': 0.0,
                'rapl_pkg_j': 0.0, 'perf_pkg_j': 0.0, 'max_duration_s': 0.0,
                'avg_cpu_pct': 0.0, 'tdp_w': 0.0, 'tdp_default': True,
                'cpu_model': 'n/a', 'method': 'n/a'}

    # Workers within a stage run in parallel, so the stage lasts as long as the
    # slowest one.
    max_duration = max((s.get(K_DUR, 0.0) for s in stats), default=0.0)

    # The dynamic share is attributable per worker, so it sums.
    dynamic_j = sum(s.get(K_PSU, 0.0) for s in stats)

    # The idle floor belongs to the host, so it is counted once over the stage
    # duration rather than once per worker.
    p_idle_w = max((s.get(K_PSU_IDLE_W, 0.0) for s in stats), default=0.0)
    idle_j = p_idle_w * max_duration

    # RAPL and perf are package-wide counters: co-located workers all read the
    # same value, so take a representative maximum instead of summing.
    rapl_j = max((s.get(K_RAPL, 0.0) for s in stats), default=0.0)
    perf_j = max((s.get(K_PERF, 0.0) for s in stats), default=0.0)

    return {
        'stage': stage,
        'workers': n,
        'psutil_dynamic_j': dynamic_j,
        'psutil_idle_j': idle_j,
        'psutil_pkg_j': dynamic_j + idle_j,
        'rapl_pkg_j': rapl_j,
        'perf_pkg_j': perf_j,
        'max_duration_s': max_duration,
        'avg_cpu_pct': sum(s.get(K_CPU, 0.0) for s in stats) / n,
        'tdp_w': max((s.get(K_TDP, 0.0) for s in stats), default=0.0),
        'tdp_default': any(s.get(K_TDP_DEFAULT, True) for s in stats),
        'cpu_model': stats[0].get(K_CPU_MODEL, 'unknown'),
        'method': stats[0].get(K_METHOD, 'n/a'),
    }


def summarize_by_stage(futures_by_stage):
    """Aggregate a multi-stage execution.

    Takes {stage_id: [futures]} in execution order and returns
    (per_stage_list, pipeline_summary). Stages run in sequence, so the pipeline
    sums both energy and the per-stage max durations.
    """
    per_stage = [summarize(futs, stage) for stage, futs in futures_by_stage.items()]
    if not per_stage:
        return [], summarize([], PIPELINE_STAGE)

    total = {
        'stage': PIPELINE_STAGE,
        'workers': max(s['workers'] for s in per_stage),
        'psutil_dynamic_j': sum(s['psutil_dynamic_j'] for s in per_stage),
        'psutil_idle_j': sum(s['psutil_idle_j'] for s in per_stage),
        'psutil_pkg_j': sum(s['psutil_pkg_j'] for s in per_stage),
        'rapl_pkg_j': sum(s['rapl_pkg_j'] for s in per_stage),
        'perf_pkg_j': sum(s['perf_pkg_j'] for s in per_stage),
        'max_duration_s': sum(s['max_duration_s'] for s in per_stage),
        'avg_cpu_pct': sum(s['avg_cpu_pct'] for s in per_stage) / len(per_stage),
        'tdp_w': max(s['tdp_w'] for s in per_stage),
        'tdp_default': any(s['tdp_default'] for s in per_stage),
        'cpu_model': per_stage[0]['cpu_model'],
        'method': per_stage[0]['method'],
    }
    return per_stage, total


def print_summary(app, cfg, summ, wall, repeat, repeats):
    flag = ' [TDP=DEFAULT]' if summ.get('tdp_default') else ''
    print(f"[{app}/{summ.get('stage', PIPELINE_STAGE)}] rep {repeat}/{repeats} "
          f"workers={summ['workers']:<3} mem={cfg.get('memory')} wall={wall:6.2f}s "
          f"dur={summ['max_duration_s']:6.2f}s "
          f"E_psutil={summ['psutil_pkg_j']:8.2f}J "
          f"(dyn={summ['psutil_dynamic_j']:7.2f} idle={summ['psutil_idle_j']:7.2f}) "
          f"E_rapl={summ['rapl_pkg_j']:8.2f}J "
          f"cpu={summ['avg_cpu_pct']:4.1f}% [{summ['method']}]{flag}")


def append_raw(path, app, cfg, summ, wall, repeat, units=1, stage='ALL'):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    row = {
        'app': app, 'stage': stage, 'workers': summ['workers'],
        'memory': cfg.get('memory'), 'work_units': units, 'repeat': repeat,
        'wall_s': round(wall, 3),
        'max_duration_s': round(summ['max_duration_s'], 3),
        'psutil_dynamic_j': round(summ.get('psutil_dynamic_j', 0.0), 3),
        'psutil_idle_j': round(summ.get('psutil_idle_j', 0.0), 3),
        'psutil_pkg_j': round(summ['psutil_pkg_j'], 3),
        'rapl_pkg_j': round(summ['rapl_pkg_j'], 3),
        'perf_pkg_j': round(summ.get('perf_pkg_j', 0.0), 3),
        'avg_cpu_pct': round(summ['avg_cpu_pct'], 2),
        'tdp_w': summ.get('tdp_w', ''), 'tdp_default': summ.get('tdp_default', ''),
        'cpu_model': summ.get('cpu_model', ''), 'method': summ.get('method', 'n/a'),
        'tstamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RAW_HEADER, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _std(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


def _num(x, default=-1.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def write_averaged(raw_path, avg_path):
    """Rebuild the averaged CSV from the raw one, grouped by (app, stage, workers, memory)."""
    if not os.path.exists(raw_path):
        return

    def col(items, name):
        return [v for v in (_num(r.get(name), None) for r in items) if v is not None]

    groups = {}
    # utf-8-sig tolerates a leading BOM, which Excel writes if the CSV is opened
    # and saved there. Without it the first field name parses as '﻿"app"'
    # and every lookup by column name fails.
    with open(raw_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames[0] != 'app':
            raise SystemExit(
                f"{raw_path}: header missing, should look like the following:\n"
                f"{','.join(RAW_HEADER)}"
            )
        for row in reader:
            key = (row['app'], row['stage'], row['workers'], row['memory'])
            groups.setdefault(key, []).append(row)

    rows = []
    for (app, stage, workers, memory), items in groups.items():
        psu = col(items, 'psutil_pkg_j')
        rapl = col(items, 'rapl_pkg_j')
        dur = col(items, 'max_duration_s')
        psu_m, psu_s = _mean(psu), _std(psu)
        rapl_m = _mean(rapl)
        # work_units normalises apps whose total work grows with the worker
        # count, so J/unit stays comparable across configurations.
        units = _num(items[0].get('work_units'), 1.0) or 1.0
        rows.append({
            'app': app, 'stage': stage, 'workers': workers, 'memory': memory,
            'work_units': units, 'n': len(items),
            'psutil_pkg_j_mean': round(psu_m, 3),
            'psutil_pkg_j_std': round(psu_s, 3),
            'psutil_pkg_j_cv_pct': round(100 * psu_s / psu_m, 2) if psu_m else 0.0,
            'psutil_dynamic_j_mean': round(_mean(col(items, 'psutil_dynamic_j')), 3),
            'psutil_idle_j_mean': round(_mean(col(items, 'psutil_idle_j')), 3),
            'rapl_pkg_j_mean': round(rapl_m, 3),
            'rapl_pkg_j_std': round(_std(rapl), 3),
            'perf_pkg_j_mean': round(_mean(col(items, 'perf_pkg_j')), 3),
            'psutil_j_per_unit': round(psu_m / units, 3),
            'rapl_j_per_unit': round(rapl_m / units, 3),
            # Cross-check between the modelled and the measured mechanism. A gap
            # that varies with configuration means the fitted power-model
            # constants are absorbing a systematic error. Zero while RAPL is
            # unavailable, since there is nothing to compare against.
            'psutil_vs_rapl_pct': (round(100 * (psu_m - rapl_m) / rapl_m, 2)
                                   if rapl_m else 0.0),
            'max_duration_s_mean': round(_mean(dur), 3),
            'max_duration_s_std': round(_std(dur), 3),
            'wall_s_mean': round(_mean(col(items, 'wall_s')), 3),
            'avg_cpu_pct_mean': round(_mean(col(items, 'avg_cpu_pct')), 2),
            'tdp_w': items[0].get('tdp_w', ''),
            'tdp_default': items[0].get('tdp_default', ''),
            'cpu_model': items[0].get('cpu_model', ''),
            'method': items[0].get('method', ''),
        })

    # Per-stage rows first, the ALL row last, then by memory and worker count.
    rows.sort(key=lambda r: (r['app'], r['stage'] == PIPELINE_STAGE, r['stage'],
                             _num(r['memory']), _num(r['workers'])))

    os.makedirs(os.path.dirname(os.path.abspath(avg_path)), exist_ok=True)
    with open(avg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=AVG_HEADER, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def profile_map(app, func, make_iterdata, config_space,
                backend='localhost', storage='localhost',
                csv_path=None, avg_path=None, reduce_fn=None, repeats=1,
                work_units=None):
    """Run 'func' for each configuration in 'config_space'.

    Each config dict needs {'workers': N} and optionally {'memory': MB}. Every
    repeat appends a raw row; the averaged file is rebuilt at the end.
    """
    import lithops
    raw_path = csv_path or DEFAULT_RAW
    avg_path = avg_path or DEFAULT_AVG
    print(f"== Profiling '{app}' on backend '{backend}' (repeats={repeats}) ==")
    for cfg in config_space:
        workers = cfg['workers']
        units = work_units(workers) if work_units else 1
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
            summ = summarize(futs, stage=PIPELINE_STAGE)
            print_summary(app, cfg, summ, wall, r, repeats)
            append_raw(raw_path, app, cfg, summ, wall, r, units)
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
                     csv_path=None, avg_path=None, repeats=1, work_units=None):
    """Like profile_map but for multi-stage workflows.

    run_fn(fexec, workers) returns either {stage_id: [futures]}, which gives one
    row per stage plus an 'ALL' row, or a flat [futures] list, recorded as 'ALL'
    only. Prefer the dict form: stage-level allocation needs the breakdown.
    """
    import lithops
    raw_path = csv_path or DEFAULT_RAW
    avg_path = avg_path or DEFAULT_AVG
    print(f"== Profiling '{app}' (pipeline) on backend '{backend}' (repeats={repeats}) ==")
    for cfg in config_space:
        workers = cfg['workers']
        units = work_units(workers) if work_units else 1
        memory = cfg.get('memory')
        for r in range(1, repeats + 1):
            kwargs = dict(backend=backend, storage=storage)
            if memory:
                kwargs['runtime_memory'] = memory
            fexec = lithops.FunctionExecutor(**kwargs)
            t0 = time.time()
            produced = run_fn(fexec, workers)
            wall = time.time() - t0

            if isinstance(produced, dict):
                per_stage, total = summarize_by_stage(produced)
            else:
                per_stage, total = [], summarize(produced, PIPELINE_STAGE)

            for summ in per_stage:
                print_summary(app, cfg, summ, wall, r, repeats)
                append_raw(raw_path, app, cfg, summ, wall, r, units)

            # configured workers, not total invocations
            total['workers'] = workers
            print_summary(app, cfg, total, wall, r, repeats)
            append_raw(raw_path, app, cfg, total, wall, r, units)
            fexec.clean()
    write_averaged(raw_path, avg_path)
    print(f"Raw:      {raw_path}")
    print(f"Averaged: {avg_path}")
