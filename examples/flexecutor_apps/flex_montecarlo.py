#!/usr/bin/env python3
"""
Monte Carlo Pi as a Flexecutor app (single-stage DAG), with energy profiling.

Purpose: show that the EnergyManager measures a workload run *through Flexecutor*
(not plain Lithops). Flexecutor runs each Stage via Lithops FunctionExecutor.map,
so the stage functions still pass through the worker handler + EnergyManager, and
the per-worker energy is exposed by StageFuture.stats (read here by flex_energy).

This app is compute-only: no FlexData inputs/outputs, so it needs no object-storage
bucket. Parallelism comes from StageConfig.workers -> N worker invocations, each
computing TOTAL_POINTS/N samples.

Prerequisites:
  - pip install .            (Flexecutor, from its repo)
  - Lithops configured with localhost as the default backend/storage, e.g. a
    ~/.lithops/config with:
        lithops:
            backend: localhost
            storage: localhost

    python examples/flexecutor_apps/flex_montecarlo.py
"""
from flexecutor.workflow.stage import Stage
from flexecutor.workflow.dag import DAG
from flexecutor.workflow.executor import DAGExecutor
from flexecutor.utils.dataclass import StageConfig
from flexecutor.utils.utils import flexorchestrator
from flexecutor.workflow.stagecontext import StageContext

from flex_energy import record_run

TOTAL_POINTS = 100_000_000
REPEATS = 5
WORKER_COUNTS = (1, 2, 4, 8)
MEMORY = 1024


def mc_stage(ctx: StageContext):
    """Monte Carlo Pi worker: count points inside the quarter circle (pure CPU)."""
    import random
    n = ctx.get_param("points_per_worker")
    inside = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def build_executor(workers):
    """Build a single-stage DAG configured for a given worker count."""
    stage = Stage(
        stage_id="montecarlo",
        func=mc_stage,
        inputs=[],
        outputs=[],
        params={"points_per_worker": TOTAL_POINTS // workers},
    )
    stage.resource_config = StageConfig(cpu=1, workers=workers, memory=MEMORY)
    dag = DAG("montecarlo_pi")
    dag.add_stage(stage)
    return DAGExecutor(dag=dag)


@flexorchestrator(bucket="")
def main():
    import time
    for workers in WORKER_COUNTS:
        for rep in range(1, REPEATS + 1):
            executor = build_executor(workers)
            t0 = time.time()
            futures = executor.execute()          # {stage_id: StageFuture}
            wall = time.time() - t0
            record_run("flex_montecarlo_pi", workers, MEMORY, rep, futures, wall)
            executor.shutdown()


if __name__ == "__main__":
    main()