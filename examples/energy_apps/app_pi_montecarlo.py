#!/usr/bin/env python3
"""
App 1/4 (Inigo Arriazu): Monte Carlo Pi.

Embarrassingly parallel computation with no inter-worker 
communication. Each worker computes random point evaluations for π estimation. 
Pure CPU workload. The total number of the points is 100M, this number is split 
between the number of workers.

    python examples/energy_apps/app_pi_montecarlo.py
"""
from energy_report import profile_map

TOTAL_POINTS = 100_000_000


def mc_worker(n_points):
    """Counts points inside the quarter circle (pure CPU)."""
    import random
    inside = 0
    for _ in range(n_points):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def make_iterdata(workers):
    """Splits TOTAL_POINTS equitatively between the workers."""
    per = TOTAL_POINTS // workers
    return [per] * workers


def estimate_pi(results):
    total_inside = sum(results)
    return f"pi ~= {4.0 * total_inside / TOTAL_POINTS:.6f}"


if __name__ == '__main__':
    # Local: sweep workers only (memory isn't enforced on localhost).
    # For AWS/K8s, add more memory values to the tuple below.
    MEMORY = [1024]
    config_space = [
        {'workers': w, 'memory': m}
        for m in MEMORY
        for w in (1, 2, 4, 8)
    ]
    profile_map('montecarlo_pi', mc_worker, make_iterdata, config_space,
                reduce_fn=estimate_pi)