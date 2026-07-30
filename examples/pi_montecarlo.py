#!/usr/bin/env python3
"""
CPU energy-monitoring demo for Lithops (localhost backend).

Runs a CPU-bound workload (Monte Carlo estimation of Pi) at increasing sizes
as parallel Lithops tasks. Each task goes through the worker handler, so the
EnergyManager (RAPL + psutil) and SystemMonitor measure it. The script reads
the per-invocation CPU and energy fields from each future's stats and prints a
table.

Run from the repo root with the virtualenv active:

    python examples/energy_demo.py

No cloud config needed (backend/storage = 'localhost'). RAPL needs read access
to /sys/class/powercap/... (usually root); otherwise the psutil modeled
estimate is used automatically.
"""

import time
import lithops


def monte_carlo_pi(n_samples):
    """Estimate Pi by sampling points in the unit square (pure CPU)."""
    import random
    inside = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return {'n_samples': n_samples, 'pi': 4.0 * inside / n_samples}


def _g(stats, key, default=0.0):
    return stats.get(key, default)


def print_table(workloads, futures):
    print("\n" + "=" * 82)
    print("CPU / ENERGY PER INVOCATION")
    print("=" * 82)
    print(f"{'samples':>12} {'dur(s)':>8} {'cpu%':>7} {'user(s)':>8} {'sys(s)':>8} "
          f"{'RAPL(J)':>10} {'psutil(J)':>10}")
    print("-" * 82)
    for n, fut in zip(workloads, futures):
        s = getattr(fut, 'stats', {}) or {}
        rapl = _g(s, 'worker_func_rapl_energy_pkg') if s.get('worker_func_rapl_available') else None
        print(f"{n:>12,} "
              f"{_g(s,'worker_func_energy_duration'):>8.3f} "
              f"{_g(s,'worker_func_psutil_avg_cpu_percent'):>7.1f} "
              f"{_g(s,'worker_func_cpu_user_time'):>8.3f} "
              f"{_g(s,'worker_func_cpu_system_time'):>8.3f} "
              f"{('%.3f' % rapl) if rapl is not None else 'n/a':>10} "
              f"{_g(s,'worker_func_psutil_energy_pkg'):>10.3f}")
    print("=" * 82)
    print("method:", (getattr(futures[0], 'stats', {}) or {}).get('worker_func_energy_method_used', 'n/a'),
          "| cpu%: media del %CPU del proceso (muestreo periodico), normalizada [0,100]")



def main():
    workloads = [20_000_000, 50_000_000, 100_000_000, 200_000_000]
    print(f"Lanzando {len(workloads)} tareas CPU-bound en backend 'localhost'...")
    t0 = time.time()
    fexec = lithops.FunctionExecutor(backend='localhost', storage='localhost')
    futures = fexec.map(monte_carlo_pi, workloads)
    results = fexec.get_result(fs=futures)
    print(f"\nHecho en {time.time()-t0:.2f}s. Estimaciones de Pi:")
    for r in results:
        print(f"  {r['n_samples']:>12,} samples -> pi ~= {r['pi']:.5f}")
    print_table(workloads, futures)
    fexec.clean()


if __name__ == '__main__':
    main()