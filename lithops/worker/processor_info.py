#
# (C) Copyright IBM Corp. 2020
# (C) Copyright Cloudlab URV 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Processor identification for the energy monitors.

Why this module exists
----------------------
The psutil power model needs a TDP reference. Without a reliable processor
string the model silently falls back to a generic default, which means the
single most important input of the model is a guess. This module resolves the
processor identity from the most trustworthy source available on each platform
and caches it, so the answer is deterministic within a runtime container.

Adapted from Green-Computing-Lithops/lithops_fork
(``lithops/worker/processor_info.py``). The public interface is deliberately
unchanged -- same three functions, same ``info`` keys, same ``worker_processor_*``
call-status keys -- so profiles remain comparable with that repo. The
implementation is a rewrite:

1. The result is cached at module level. Processor identity cannot change
   within a container, and the original re-ran ``lscpu`` and a ``curl`` on
   every single invocation. That subprocess activity is charged to the very
   process whose CPU time we are measuring, so it biased the energy figures it
   was meant to support.
2. Every external command is guarded by ``shutil.which`` and a timeout, so a
   missing ``lscpu`` degrades instead of logging an exception per invocation,
   and no probe can hang the worker.
3. ARM implementer/part decoding, which the original lacked. Graviton exposes
   no ``model name`` line in /proc/cpuinfo, so without this the processor is
   unidentifiable on the primary experimental platform and the TDP always
   falls back.
4. TDP resolution (``resolve_tdp``) against a table with per-entry provenance
   and an ``is_default`` flag. The original had no TDP logic at all.
5. The EC2 IMDS probe was removed rather than fixed. It returns nothing on
   either platform in use -- on Lambda 169.254.169.254 is not the metadata
   service, and the local cluster has none -- so it was cost without evidence.
   The original issued an unbounded ``curl`` there, which stalls a worker.
6. Windows uses PowerShell CIM instead of ``wmic`` (removed by default on
   Windows 11 24H2+). macOS support was dropped: no worker runs there.
"""

import json
import logging
import os
import platform
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Processor identity is constant for the lifetime of a runtime container.
_CACHED_INFO = None

# Seconds. Every external command is bounded; a probe that hangs would stall
# the worker and inflate the measured duration it is meant to support.
_CMD_TIMEOUT = 2.0


def _is_lambda():
    return bool(os.environ.get("AWS_LAMBDA_RUNTIME_API"))


def _run(cmd, timeout=_CMD_TIMEOUT):
    """Run a command list, returning stdout or None. Never raises."""
    if not cmd or shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.check_output(
            cmd, timeout=timeout, stderr=subprocess.DEVNULL
        )
        return out.decode(errors="replace")
    except Exception as e:
        logger.debug(f"Command {cmd[0]} failed: {e}")
        return None


def _brand_from_name(name):
    if not name:
        return None
    low = name.lower()
    if "intel" in low:
        return "Intel"
    if "amd" in low:
        return "AMD"
    if "graviton" in low or "neoverse" in low or "arm" in low:
        return "ARM"
    if "apple" in low:
        return "Apple"
    first = name.split()[0]
    return first if first and first[0].isupper() else None


def _collect_linux(info):
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
    except Exception as e:
        logger.debug(f"Cannot read /proc/cpuinfo: {e}")
        cpuinfo = ""

    if "hypervisor" in cpuinfo:
        info["is_virtual"] = True

    m = re.search(r"model name\s*:\s*(.*)", cpuinfo)
    if m:
        info["processor_name"] = m.group(1).strip()
    else:
        # ARM (Graviton) exposes implementer/part instead of a model name.
        impl = re.search(r"CPU implementer\s*:\s*(\S+)", cpuinfo)
        part = re.search(r"CPU part\s*:\s*(\S+)", cpuinfo)
        if impl and part:
            info["processor_name"] = _decode_arm(impl.group(1), part.group(1))

    info["processor_brand"] = _brand_from_name(info["processor_name"])

    lscpu = _run(["lscpu"])
    if not lscpu:
        return

    sockets = 1
    cores_per_socket = None
    for line in lscpu.splitlines():
        if "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
            try:
                info["threads"] = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "Core(s) per socket:" in line:
            try:
                cores_per_socket = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "Socket(s):" in line:
            try:
                sockets = int(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "CPU max MHz:" in line:
            try:
                info["frequency"] = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "CPU min MHz:" in line:
            try:
                info["min_frequency"] = float(line.split(":")[1].strip())
            except ValueError:
                pass
        elif "L3 cache:" in line:
            info["cache_size"] = line.split(":", 1)[1].strip()
        elif "Model name:" in line and not info["processor_name"]:
            info["processor_name"] = line.split(":", 1)[1].strip()
            info["processor_brand"] = _brand_from_name(info["processor_name"])
        elif "Vendor ID:" in line and not info["processor_brand"]:
            info["processor_brand"] = _brand_from_name(line.split(":", 1)[1].strip())

    if cores_per_socket:
        info["cores"] = cores_per_socket * sockets


# Minimal ARM implementer/part decode, enough to recognise Graviton parts.
_ARM_PARTS = {
    ("0x41", "0xd0c"): "ARM Neoverse-N1 (AWS Graviton2)",
    ("0x41", "0xd40"): "ARM Neoverse-V1 (AWS Graviton3)",
    ("0x41", "0xd4f"): "ARM Neoverse-V2 (AWS Graviton4)",
    ("0x41", "0xd49"): "ARM Neoverse-N2",
}


def _decode_arm(implementer, part):
    return _ARM_PARTS.get(
        (implementer.lower(), part.lower()), f"ARM implementer={implementer} part={part}"
    )


def _collect_windows(info):
    """
    Processor identity on native Windows.

    Kept because the localhost backend does run natively on Windows, where
    neither RAPL (/sys/class/powercap) nor perf exists -- the psutil power
    model is the ONLY mechanism available, so it is exactly the platform where
    a resolved processor name matters most.

    Uses PowerShell CIM rather than `wmic`. WMIC has been deprecated since
    Windows 10 21H1 and is removed by default on Windows 11 24H2/25H2, where
    it is only available as a Feature on Demand -- so a wmic-only
    implementation silently yields nothing on a current machine. wmic is
    retained as a fallback for older hosts.
    """
    parsed = False
    out = _run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Processor | "
            "Select-Object -First 1 Name,NumberOfCores,"
            "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Csv -NoTypeInformation",
        ],
        timeout=8.0,  # PowerShell start-up is slow; this runs once per container
    )
    if out:
        rows = [r for r in out.strip().splitlines() if r.strip()]
        if len(rows) >= 2:
            header = [h.strip().strip('"') for h in rows[0].split(",")]
            values = [v.strip().strip('"') for v in rows[-1].split(",")]
            parsed = _apply_windows_row(info, dict(zip(header, values)))

    if parsed:
        return

    out = _run(
        [
            "wmic", "cpu", "get",
            "name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed",
            "/format:csv",
        ]
    )
    if not out:
        logger.debug("Neither PowerShell CIM nor wmic returned processor data")
        return
    rows = [r for r in out.strip().splitlines() if r.strip()]
    if len(rows) < 2:
        return
    header = [h.strip() for h in rows[0].split(",")]
    values = [v.strip() for v in rows[-1].split(",")]
    _apply_windows_row(info, dict(zip(header, values)))


def _apply_windows_row(info, row):
    """Populate `info` from a parsed Win32_Processor row. True if a name was found."""
    name = (row.get("Name") or "").strip()
    if name:
        info["processor_name"] = name
        info["processor_brand"] = _brand_from_name(name)
    for key, field, cast in (
        ("NumberOfCores", "cores", int),
        ("NumberOfLogicalProcessors", "threads", int),
        ("MaxClockSpeed", "frequency", float),
    ):
        try:
            info[field] = cast(row[key])
        except (KeyError, ValueError, TypeError):
            pass
    return bool(name)


def get_processor_info(refresh=False):
    """
    Detailed processor information. Cached: the result is computed once per
    process and reused, because it cannot change and because recomputing it
    charges CPU time to the measured function.
    """
    global _CACHED_INFO
    if _CACHED_INFO is not None and not refresh:
        return dict(_CACHED_INFO)

    info = {
        "processor_name": None,
        "processor_brand": None,
        "cores": None,
        "threads": None,
        "architecture": platform.machine(),
        "frequency": None,
        "cache_size": None,
        # Retained for interface parity with flexecutor-main's stagefuture.py,
        # which reads worker_cloud_instance_type. Never populated now: the IMDS
        # probe was removed because it returns nothing on either platform this
        # is used on (on Lambda 169.254.169.254 is not the metadata service; on
        # the local cluster there is no metadata service at all).
        "cloud_instance_type": None,
        "is_virtual": False,
        "is_lambda": _is_lambda(),
        "lambda_memory_mb": int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", 0) or 0),
        "source": "unknown",
    }

    # Linux covers Lambda and the cluster; Windows covers a native localhost
    # backend, where the psutil model is the only energy mechanism available.
    # macOS is not supported: no worker ever runs there, and an untested
    # collector is worse than an honest fallback.
    system = platform.system()
    try:
        if system == "Linux":
            _collect_linux(info)
            info["source"] = "proc_cpuinfo+lscpu"
        elif system == "Windows":
            _collect_windows(info)
            info["source"] = "win32_processor"
        else:
            logger.info(
                f"No processor collector for {system}; falling back to "
                "platform.processor(). TDP will be flagged as a default."
            )
    except Exception as e:
        logger.warning(f"Processor detection failed on {system}: {e}")

    if not info["processor_name"]:
        info["processor_name"] = platform.processor() or "Unknown"
        info["source"] = "platform_fallback"
    if not info["processor_brand"]:
        info["processor_brand"] = _brand_from_name(info["processor_name"]) or "Unknown"

    if not info["cores"] or not info["threads"]:
        try:
            import psutil

            info["cores"] = info["cores"] or psutil.cpu_count(logical=False)
            info["threads"] = info["threads"] or psutil.cpu_count(logical=True)
        except Exception:
            import multiprocessing

            info["threads"] = info["threads"] or multiprocessing.cpu_count()

    _CACHED_INFO = info
    logger.info(
        f"Processor: {info['processor_name']} ({info['processor_brand']}), "
        f"arch={info['architecture']}, cores={info['cores']}, threads={info['threads']}, "
        f"instance={info['cloud_instance_type']}, source={info['source']}"
    )
    return dict(info)


def get_processor_info_json():
    return json.dumps(get_processor_info(), indent=2)


# ---------------------------------------------------------------------------
# TDP reference resolution
# ---------------------------------------------------------------------------
# Every entry is a *package* TDP published by the vendor for that exact part.
# Provenance is recorded per entry.
#
# Matching is substring-based on the processor name, longest match first, so
# "Xeon Platinum 8259CL" wins over the generic "Xeon Platinum" family entry.
#
# Citation classes, weakest last. They are not equivalent evidence:
#   [A] vendor datasheet (Intel ARK, AMD product page)
#   [B] reseller / benchmark-database listing -- the only public figure that
#       exists for AWS custom SKUs, which have no vendor page at all
#   [C] third-party analyst estimate; the vendor has never disclosed a figure
#
# Deliberately NO family-level rows (e.g. a generic "xeon platinum"). Intel
# Platinum spans roughly 105-400W and Xeon Gold 85-205W, so a single family
# number is a guess -- and a guess matched here would be returned with
# is_default=False, i.e. indistinguishable from a resolved value. Anything not
# listed below falls through to the flagged default instead, which is the
# honest outcome.
_TDP_TABLE = [
    # (match substring, TDP watts, provenance)
    ("neoverse-v2", 130.0,
     "[C] AWS Graviton4 (Neoverse-V2) ~130W; analyst estimate, AWS has never "
     "published a TDP"),
    ("neoverse-v1", 100.0,
     "[C] AWS Graviton3 (Neoverse-V1) ~100W; press/analyst estimate, not "
     "published by AWS"),
    ("neoverse-n1", 100.0,
     "[C] AWS Graviton2 (Neoverse-N1) ~100W; reported as comparable to "
     "Graviton3. NOTE: a lower figure (~80W) is sometimes quoted -- verify "
     "against your own source before relying on this"),
    ("epyc 7r13", 280.0, "[B] AMD EPYC 7R13, AWS custom Milan (c6a); 280W"),
    ("epyc 7r32", 280.0, "[B] AMD EPYC 7R32, AWS custom Rome (c5a); 280W"),
    ("ryzen 7 5800h", 45.0,
     "[A] AMD Ryzen 7 5800H (Cezanne, Zen 3, 8C/16T mobile); AMD default TDP "
     "45W. Development host"),
    ("xeon platinum 8488c", 385.0,
     "[B] Intel Xeon Platinum 8488C, AWS custom Sapphire Rapids; 385W. "
     "DISPUTED: 350W is also reported, matching the comparable 8480+/8480C. "
     "No Intel ARK page exists for this SKU"),
    ("xeon platinum 8375c", 300.0,
     "[B] Intel Xeon Platinum 8375C (Ice Lake-SP), AWS custom; 300W"),
    ("xeon platinum 8259cl", 210.0,
     "[B] Intel Xeon Platinum 8259CL (Cascade Lake-SP), AWS custom; 210W "
     "(165W occasionally quoted)"),
    ("xeon platinum 8175m", 240.0,
     "[B] Intel Xeon Platinum 8175M (Skylake-SP), AWS custom; 240W"),
]

# Used only when nothing above matches. These are ORDER-OF-MAGNITUDE
# PLACEHOLDERS, not specifications -- there is no such thing as "the" TDP of an
# unidentified x86 server part. They exist so the model produces a number
# rather than crashing; every run that uses one is flagged is_default=True in
# the emitted fields and in the profiling CSV, so it can be excluded from any
# result that depends on absolute joules.
_DEFAULT_TDP_X86 = 125.0
_DEFAULT_TDP_ARM = 100.0


def _normalise_name(name):
    """
    Normalise a processor string before table matching.

    Intel reports itself as "Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz",
    so a needle like "xeon platinum 8259cl" is NOT a substring of the raw
    lowercased name -- the "(R)" sits between "xeon" and "platinum". Matching
    on the raw string silently defaulted every Intel host, which is precisely
    the failure this module exists to prevent, and it would have been invisible
    apart from the is_default flag.

    Strips (R)/(TM)/(C) marks and collapses whitespace. AMD and ARM strings
    carry no such marks and are unaffected.
    """
    if not name:
        return ""
    low = name.lower()
    for mark in ("(r)", "(tm)", "(c)", "®", "™"):
        low = low.replace(mark, " ")
    return " ".join(low.split())


def resolve_tdp(info=None):
    """
    Resolve a package TDP reference for the current processor.

    Returns a dict with the value, its provenance, and whether it is a
    fallback. The caller is expected to surface ``is_default`` so that a
    guessed TDP never silently becomes an experimental result.
    """
    info = info or get_processor_info()
    name = _normalise_name(info.get("processor_name"))
    arch = (info.get("architecture") or "").lower()

    for needle, watts, provenance in sorted(
        _TDP_TABLE, key=lambda e: -len(e[0])
    ):
        if needle in name:
            return {
                "tdp_w": watts,
                "tdp_source": provenance,
                "tdp_matched_on": needle,
                "is_default": False,
            }

    is_arm = "aarch64" in arch or "arm" in arch
    return {
        "tdp_w": _DEFAULT_TDP_ARM if is_arm else _DEFAULT_TDP_X86,
        "tdp_source": f"UNRESOLVED default for arch={arch!r}, name={name!r}",
        "tdp_matched_on": None,
        "is_default": True,
    }


def add_processor_info_to_task(task, call_status):
    """
    Publish processor information into the call status (and the stats file) so
    it reaches the client alongside the energy fields.
    """
    try:
        info = get_processor_info()

        call_status.add("worker_processor_info", info)
        call_status.add("worker_processor_name", info["processor_name"])
        call_status.add("worker_processor_brand", info["processor_brand"])
        call_status.add("worker_processor_cores", info["cores"])
        call_status.add("worker_processor_threads", info["threads"])
        call_status.add("worker_processor_architecture", info["architecture"])
        call_status.add("worker_processor_source", info["source"])
        if info["cloud_instance_type"]:
            call_status.add("worker_cloud_instance_type", info["cloud_instance_type"])

        stats_file = getattr(task, "stats_file", None)
        if stats_file:
            with open(stats_file, "a") as f:
                f.write(f"processor_name {info['processor_name']}\n")
                f.write(f"processor_brand {info['processor_brand']}\n")
                f.write(f"processor_cores {info['cores']}\n")
                f.write(f"processor_threads {info['threads']}\n")
                if info["frequency"]:
                    f.write(f"processor_frequency {info['frequency']}\n")
                if info["cloud_instance_type"]:
                    f.write(f"cloud_instance_type {info['cloud_instance_type']}\n")
        return info
    except Exception as e:
        logger.warning(f"Error publishing processor information: {e}")
        return None


if __name__ == "__main__":
    print(get_processor_info_json())
