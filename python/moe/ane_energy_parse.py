"""Parse a powermetrics plist + workload-probe stdout to compute J/expert-eval.

Usage:
    python python/moe/ane_energy_parse.py /tmp/power.plist <workload_log.txt>

The workload log must contain BEGIN_AT, END_AT, and ITERS lines.
"""
import datetime as dt
import plistlib
import re
import sys
from pathlib import Path


def parse_workload(p: Path):
    txt = p.read_text()
    begin = float(re.search(r"BEGIN_AT\s+(\S+)", txt).group(1))
    end = float(re.search(r"END_AT\s+(\S+)", txt).group(1))
    iters = int(re.search(r"ITERS\s+(\d+)", txt).group(1))
    return begin, end, iters


def split_plists(raw: bytes):
    """Powermetrics emits N concatenated XML plist docs separated by NUL."""
    marker = b"<?xml"
    parts = raw.split(marker)
    return [(marker + p).rstrip(b"\x00\n\r ") for p in parts if p.strip()]


def find_power(d):
    """Recursively find cpu/gpu/ane_power and package/combined_power keys."""
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("cpu_power", "gpu_power", "ane_power",
                         "package_power", "combined_power") and isinstance(v, (int, float)):
                    found[k] = float(v)
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(d)
    return found


def parse_power(p: Path, t_begin: float, t_end: float):
    raw = p.read_bytes()
    docs = split_plists(raw)
    samples = []
    for d in docs:
        try:
            samples.append(plistlib.loads(d))
        except Exception:
            pass
    if not samples:
        print("ERROR: no plist samples parsed", file=sys.stderr)
        sys.exit(2)

    print(f"parsed {len(samples)} samples")
    rows_in = []
    rows_all = []
    for s in samples:
        ts = s.get("timestamp")
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        t = ts.timestamp()
        elapsed_s = s.get("elapsed_ns", 1e9) / 1e9
        pw = find_power(s)
        row = (t, elapsed_s,
               pw.get("cpu_power", 0.0),
               pw.get("gpu_power", 0.0),
               pw.get("ane_power", 0.0),
               pw.get("combined_power", pw.get("package_power", 0.0)))
        rows_all.append(row)
        if t_begin <= t <= t_end + 2.0:
            rows_in.append(row)

    if not rows_in:
        print("ERROR: no samples within workload window", file=sys.stderr)
        print(f"  workload window: {t_begin:.2f} .. {t_end:.2f}", file=sys.stderr)
        if rows_all:
            print(f"  sample timestamps span: {rows_all[0][0]:.2f} .. {rows_all[-1][0]:.2f}",
                  file=sys.stderr)
        sys.exit(3)

    e_cpu = sum(s * cpu for _, s, cpu, gpu, ane, pkg in rows_in) / 1000.0
    e_gpu = sum(s * gpu for _, s, cpu, gpu, ane, pkg in rows_in) / 1000.0
    e_ane = sum(s * ane for _, s, cpu, gpu, ane, pkg in rows_in) / 1000.0
    e_pkg = sum(s * pkg for _, s, cpu, gpu, ane, pkg in rows_in) / 1000.0
    total_dt = sum(s for _, s, *_ in rows_in)

    return {
        "samples": len(rows_in),
        "window_s": total_dt,
        "energy_cpu_J": e_cpu,
        "energy_gpu_J": e_gpu,
        "energy_ane_J": e_ane,
        "energy_pkg_J": e_pkg,
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    plist_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2])

    begin, end, iters = parse_workload(log_path)
    print(f"workload: {iters} iters over {end - begin:.2f} s   ({iters / (end - begin):.0f}/s)")

    e = parse_power(plist_path, begin, end)
    w = e["window_s"]
    print(f"window:    {e['samples']} samples, {w:.2f} s covered")
    print(f"  CPU:  {e['energy_cpu_J']:8.2f} J   ({e['energy_cpu_J']/w*1000:7.0f} mW avg)")
    print(f"  GPU:  {e['energy_gpu_J']:8.2f} J   ({e['energy_gpu_J']/w*1000:7.0f} mW avg)")
    print(f"  ANE:  {e['energy_ane_J']:8.2f} J   ({e['energy_ane_J']/w*1000:7.0f} mW avg)")
    print(f"  PKG:  {e['energy_pkg_J']:8.2f} J   ({e['energy_pkg_J']/w*1000:7.0f} mW avg)")
    print()
    print(f"per expert-eval (over {iters} iters):")
    print(f"  CPU:  {e['energy_cpu_J'] / iters * 1e3:7.4f} mJ")
    print(f"  GPU:  {e['energy_gpu_J'] / iters * 1e3:7.4f} mJ")
    print(f"  ANE:  {e['energy_ane_J'] / iters * 1e3:7.4f} mJ")
    print(f"  PKG:  {e['energy_pkg_J'] / iters * 1e3:7.4f} mJ  (total)")


if __name__ == "__main__":
    main()
