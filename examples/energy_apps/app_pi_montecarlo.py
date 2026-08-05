#!/usr/bin/env python3
"""
App 1/4 (Inigo Arriazu): Monte Carlo Pi.

Embarrassingly parallel, pure CPU. TOTAL_POINTS points are split evenly among the
workers (more workers -> less work per worker, same total work). Each worker
counts how many points fall inside the quarter circle; pi is estimated from the
combined count.

    python examples/energy_apps/app_montecarlo.py
"""
from energy_report import profile_map

TOTAL_POINTS = 100_000_000   # as in the thesis


def mc_worker(n_points):
    """Count points inside the quarter circle (pure CPU)."""
    import random
    inside = 0
    for _ in range(n_points):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def make_iterdata(workers):
    """Split TOTAL_POINTS evenly among the workers."""
    per = TOTAL_POINTS // workers
    return [per] * workers


def estimate_pi(results):
    total_inside = sum(results)
    return f"pi ~= {4.0 * total_inside / TOTAL_POINTS:.6f}"


if __name__ == '__main__':
    # Config space: grid of workers x memory (same total work per row)
    # Local: sweep workers only (memory is not enforced on localhost).
    # For AWS/K8s, add more values to MEMORY below.
    MEMORY = [1024]
    config_space = [
        {'workers': w, 'memory': m}
        for m in MEMORY
        for w in (1, 2, 4, 8)
    ]
    profile_map('montecarlo_pi', mc_worker, make_iterdata, config_space,
                reduce_fn=estimate_pi, repeats=5)