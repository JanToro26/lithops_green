#!/usr/bin/env python3
"""Fit IDLE_FRACTION and DYNAMIC_FRACTION from an HWiNFO64 CSV log.

Model: P_pkg(U) = TDP * (IDLE_FRACTION + DYNAMIC_FRACTION * U), U in [0, 1].
Least squares on power vs. utilisation gives intercept/TDP = IDLE_FRACTION and
slope/TDP = DYNAMIC_FRACTION.

Log: HWiNFO64 sensors-only, Logging Start, ~3 min idle (the intercept is the
idle fraction, so it has to be measured), then a workload that sweeps
utilisation, then Logging Stop.

    python examples/energy_apps/calibrate_power.py Test1.CSV --tdp 45
    python examples/energy_apps/calibrate_power.py Test1.CSV --list-columns
"""
import argparse
import csv
import re
import sys

# Ordered by preference. HWiNFO localises column names, so both the Spanish and
# English forms are listed; matching is on lowercased substrings.
POWER_NEEDLES = [
    ("potencia total de cpu",),   # AMD SMU package power
    ("cpu ppt",),                 # Package Power Tracking, same scope
    ("package power",),
    ("cpu package",),
    ("core+soc", "potencia"),     # measured rails, excludes some uncore
]
USAGE_NEEDLES = [
    ("uso total de cpu",),
    ("total cpu usage",),
    ("uso núcleo (avg)",),
    ("uso nucleo (avg)",),
    # "Utilización total de la CPU" / "Total CPU Utility" is deliberately last:
    # it is frequency-scaled and can exceed 100%, so it is not the 0-1 package
    # utilisation the model is defined against.
    ("utilización total de la cpu",),
    ("total cpu utility",),
]

TIME_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}")


def _to_float(text):
    """Parse a float written with either decimal separator."""
    if text is None:
        return None
    text = text.strip().strip('"')
    if not text or text.lower() in ("n/a", "na", "-"):
        return None
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _pick(fieldnames, needle_sets):
    for needles in needle_sets:
        for name in fieldnames:
            low = name.lower()
            if all(n in low for n in needles):
                return name
    return None


def load(path):
    """Return (rows, fieldnames), dropping HWiNFO's trailer rows."""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.DictReader(f))
    # HWiNFO appends a repeat of the header and a system-description row. Rather
    # than dropping a fixed count, keep only rows whose Time field is a clock
    # reading, which is robust to however many trailers a version writes.
    time_key = next((k for k in (rows[0].keys() if rows else []) if k and k.strip().lower() == "time"), None)
    if time_key:
        rows = [r for r in rows if TIME_RE.match((r.get(time_key) or "").strip())]
    fields = [k for k in (rows[0].keys() if rows else []) if k]
    return rows, fields


def fit(pairs):
    """Least squares. Returns (intercept, slope, r_squared)."""
    n = len(pairs)
    mx = sum(u for u, _ in pairs) / n
    my = sum(p for _, p in pairs) / n
    sxx = sum((u - mx) ** 2 for u, _ in pairs)
    sxy = sum((u - mx) * (p - my) for u, p in pairs)
    if sxx == 0:
        sys.exit("All samples share one utilisation; the log has no load sweep.")
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((p - (intercept + slope * u)) ** 2 for u, p in pairs)
    ss_tot = sum((p - my) ** 2 for _, p in pairs)
    return intercept, slope, (1 - ss_res / ss_tot if ss_tot else 0.0)


def pairs_for(rows, usage_col, power_col):
    out = []
    for r in rows:
        u, p = _to_float(r.get(usage_col)), _to_float(r.get(power_col))
        if u is not None and p is not None and p > 0:
            out.append((u / 100.0, p))
    return out


def binned_table(pairs, intercept, slope, width=10):
    """Mean measured vs fitted power per utilisation band.

    Curvature here means the linear form is wrong, not the constants.
    """
    buckets = {}
    for u, p in pairs:
        b = min(int(u * 100 // width) * width, 100 - width)
        buckets.setdefault(b, []).append(p)
    print(f"  {'band':>9} {'n':>5} {'measured':>10} {'fitted':>9} {'resid':>8}")
    for b in sorted(buckets):
        vals = buckets[b]
        mid = (b + width / 2) / 100.0
        meas = sum(vals) / len(vals)
        pred = intercept + slope * mid
        print(f"  {b:3}-{b+width-1:3}% {len(vals):5} {meas:9.2f}W {pred:8.2f}W "
              f"{meas - pred:+7.2f}W")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--tdp", type=float, default=45.0)
    ap.add_argument("--power-col")
    ap.add_argument("--usage-col")
    ap.add_argument("--idle-below", type=float, default=5.0,
                    help="samples under this CPU%% are treated as idle (default 5)")
    ap.add_argument("--max-util", type=float, default=100.0,
                    help="fit only samples at or below this CPU%%. Package power "
                         "saturates near the sustained power limit, so a fit over "
                         "the full range is dragged by a region the workloads never "
                         "enter. Restrict this to the utilisation your experiments "
                         "actually reach (default 100)")
    ap.add_argument("--list-columns", action="store_true",
                    help="print candidate power and usage columns, then exit")
    ap.add_argument("--compare", action="store_true",
                    help="fit every candidate power column, not just the best")
    args = ap.parse_args()

    rows, fields = load(args.csv_path)
    if not rows:
        sys.exit(f"{args.csv_path}: no sample rows found.")

    if args.list_columns:
        print("power candidates:")
        for c in fields:
            if "[W]" in c:
                print("   ", c)
        print("\nusage candidates:")
        for c in fields:
            if "[%]" in c and any(k in c.lower() for k in ("cpu", "uso", "carga", "usage")):
                print("   ", c)
        return

    usage_col = args.usage_col or _pick(fields, USAGE_NEEDLES)
    power_col = args.power_col or _pick(fields, POWER_NEEDLES)
    if not usage_col or not power_col:
        sys.exit("Could not identify the columns. Re-run with --list-columns "
                 "and pass --power-col / --usage-col.")

    print(f"samples      : {len(rows)}")
    print(f"usage column : {usage_col}")
    print(f"power column : {power_col}")

    if args.compare:
        print("\nall candidate power columns:")
        for needles in POWER_NEEDLES:
            col = _pick(fields, [needles])
            if not col:
                continue
            pr = pairs_for(rows, usage_col, col)
            if len(pr) < 10:
                continue
            b, m, r2 = fit(pr)
            print(f"  {col:<45} idle={b/args.tdp:.4f} dyn={m/args.tdp:.4f} R2={r2:.4f}")

    pairs = pairs_for(rows, usage_col, power_col)
    all_pairs = pairs
    if args.max_util < 100.0:
        pairs = [x for x in pairs if x[0] * 100 <= args.max_util]
        print(f"restricted   : U <= {args.max_util:.0f}%  "
              f"({len(pairs)} of {len(all_pairs)} samples)")
    if len(pairs) < 10:
        sys.exit(f"Only {len(pairs)} usable samples.")

    intercept, slope, r2 = fit(pairs)
    us = [u * 100 for u, _ in pairs]
    idle = [p for u, p in pairs if u * 100 < args.idle_below]

    print(f"utilisation  : {min(us):.1f}% .. {max(us):.1f}%")
    print()
    print(f"  P(U) = {intercept:.2f} + {slope:.2f} * U     R^2 = {r2:.4f}")
    print()
    print(f"  IDLE_FRACTION    = {intercept / args.tdp:.4f}   (currently 0.15)")
    print(f"  DYNAMIC_FRACTION = {slope / args.tdp:.4f}   (currently 0.85)")
    print()
    binned_table(pairs, intercept, slope)

    print()
    if idle:
        measured = sum(idle) / len(idle)
        print(f"  measured idle power : {measured:.2f} W over {len(idle)} samples "
              f"below {args.idle_below}% CPU")
        print(f"  fitted intercept    : {intercept:.2f} W")
        if abs(measured - intercept) > 0.15 * max(measured, 1e-9):
            print("  These disagree by more than 15%, so the linear extrapolation "
                  "does not reach the real idle point.")
    else:
        print(f"  NO IDLE ANCHOR: the log contains no sample below "
              f"{args.idle_below}% CPU (lowest is {min(us):.1f}%). The intercept "
              f"is extrapolated, not measured. Re-log with a few minutes of "
              f"genuine idle before starting the workload.")

    if r2 < 0.9:
        print(f"\n  WARNING: R^2 = {r2:.3f}. Utilisation alone does not explain "
              f"package power on this part -- frequency and boost state matter "
              f"too. No choice of these two constants will fix that; the shape "
              f"of the model is the limitation. Worth reporting as a result.")

    print("\nSet both in lithops/worker/energymonitor_psutil.py (lines 46-47).")


if __name__ == "__main__":
    main()
