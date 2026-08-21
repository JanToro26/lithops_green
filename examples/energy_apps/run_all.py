#!/usr/bin/env python3
"""
Run the four parallel apps in sequence to fill the profiling CSV in one command.

Each app already repeats every configuration internally (repeats=5), so a single
invocation of this script produces the full profiling set.

    python examples/energy_apps/run_all.py
"""
import runpy
import sys
import os
import traceback

APPS = ['app_pi_montecarlo.py', 'app_titanic.py', 'app_ml_ensemble.py', 'app_video.py']


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for app in APPS:
        path = os.path.join(here, app)
        print(f"\n########## Running {app} ##########")
        try:
            runpy.run_path(path, run_name='__main__')
        except KeyboardInterrupt:
            raise
        except BaseException:
            print(f"!! {app} failed:", file=sys.stderr)
            traceback.print_exc()
        except Exception:
            print(f"!! {app} failed:", file=sys.stderr)
            traceback.print_exc()
    print("\nAll apps finished.")
    


if __name__ == '__main__':
    main()