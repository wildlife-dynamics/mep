#!/usr/bin/env python3
"""Parse otel_traces.jsonl and print a clean per-task timing summary."""

import json
import sys
from datetime import datetime
from pathlib import Path


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    return f"{n / 1024**2:.0f} MB"


def fmt_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.0f}s"


def _collect_machine_spec(results_dir: Path) -> dict:
    """Gather machine info at parse time — mirrors sitecustomize._write_machine_spec."""
    import os
    import platform
    import sys

    spec = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil

        spec["cpu_count_physical"] = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory()
        spec["ram_gb"] = round(mem.total / (1024**3), 1)
        spec["ram_available_gb"] = round(mem.available / (1024**3), 1)
        try:
            freq = psutil.cpu_freq(percpu=False)
            if freq and freq.max:
                spec["cpu_freq_max_ghz"] = round(freq.max / 1000, 2)
        except Exception:
            pass
        try:
            disk = psutil.disk_usage(str(results_dir))
            spec["disk_free_gb"] = round(disk.free / (1024**3), 1)
        except Exception:
            pass
    except ImportError:
        try:
            if platform.system() == "Darwin":
                import subprocess

                mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
                spec["ram_gb"] = round(mem / (1024**3), 1)
            elif platform.system() == "Linux":
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            spec["ram_gb"] = round(int(line.split()[1]) / (1024**2), 1)
                            break
        except Exception:
            pass
        try:
            import shutil

            disk = shutil.disk_usage(str(results_dir))
            spec["disk_free_gb"] = round(disk.free / (1024**3), 1)
        except Exception:
            pass
    return spec


def print_machine_spec(traces_path: str, run_start: datetime | None = None) -> dict:
    """Print machine and environment info. Returns the spec dict for later use."""
    spec_path = Path(traces_path).parent / "machine_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
    else:
        spec = _collect_machine_spec(spec_path.parent)

    rows = []
    if run_start:
        rows.append(("run started", run_start.strftime("%Y-%m-%d %H:%M:%S UTC")))
    if "hostname" in spec:
        rows.append(("machine name", spec["hostname"]))
    if "os" in spec:
        rows.append(("operating system", spec["os"]))
    if "python" in spec:
        rows.append(("python version", spec["python"]))

    cpu = spec.get("cpu", "")
    logical = spec.get("cpu_count_logical")
    physical = spec.get("cpu_count_physical")
    cpu_freq = spec.get("cpu_freq_max_ghz")
    if cpu or logical or cpu_freq:
        parts = []
        if logical and physical:
            parts.append(f"{logical} logical, {physical} physical cores")
        elif logical:
            parts.append(f"{logical} cores")
        if cpu_freq:
            parts.append(f"{cpu_freq} GHz max")
        cores = f"  ({', '.join(parts)})" if parts else ""
        rows.append(("processor", f"{cpu}{cores}"))

    if "ram_gb" in spec:
        ram = f"{spec['ram_gb']} GB total"
        if "ram_available_gb" in spec:
            ram += f",  {spec['ram_available_gb']} GB free at start"
        rows.append(("memory (RAM)", ram))

    if "disk_free_gb" in spec:
        rows.append(("disk free at start", f"{spec['disk_free_gb']} GB"))

    if not rows:
        return spec

    key_width = max(len(k) for k, _ in rows)
    print("Machine:")
    for key, val in rows:
        print(f"  {key:<{key_width}}  {val}")

    print()
    return spec


def main(traces_path: str) -> int:
    spans = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spans.append(json.loads(line))

    # Index spans by span_id for child lookups
    by_id = {s["context"]["span_id"]: s for s in spans}

    # Root span is the one with no parent (the CLI wrapper span)
    root = next((s for s in spans if s.get("parent_id") is None), None)
    if root is None:
        print("Could not find root span.", file=sys.stderr)
        return 1

    root_id = root["context"]["span_id"]

    # Top-level task spans are direct children of root with method="call"
    tasks = [s for s in spans if s.get("parent_id") == root_id and s.get("attributes", {}).get("method") == "call"]
    tasks.sort(key=lambda s: s["start_time"])

    if not tasks:
        print("No task spans found.", file=sys.stderr)
        return 1

    # For each task span, find its inner function span (child with func.__name__)
    func_name_by_task_id: dict[str, str] = {}
    for s in spans:
        pid = s.get("parent_id")
        fn = s.get("attributes", {}).get("func.__name__")
        if pid and fn and pid in by_id:
            parent = by_id[pid]
            if parent.get("attributes", {}).get("method") == "call":
                func_name_by_task_id[pid] = fn

    workflow_start = parse_ts(tasks[0]["start_time"])

    name_width = max(len(t["name"]) for t in tasks)
    func_width = max(
        (len(func_name_by_task_id.get(t["context"]["span_id"], "")) for t in tasks),
        default=0,
    )
    line_width = max(name_width + func_width + 45, 80)

    total = 0.0
    failed = 0

    for t in tasks:
        start = parse_ts(t["start_time"])
        end_ts = t.get("end_time") or t.get("start_time")
        end = parse_ts(end_ts)
        duration = (end - start).total_seconds()
        offset = (start - workflow_start).total_seconds()
        total += duration

        status_code = t.get("status", {}).get("status_code", "UNSET")
        passed = status_code != "ERROR"
        if not passed:
            failed += 1

        label = "PASSED" if passed else "FAILED"
        fn = func_name_by_task_id.get(t["context"]["span_id"], "")
        fn_str = f"({fn})" if fn else ""
        task_str = f"{t['name']:<{name_width}}  {fn_str:<{func_width + 2}}"
        timing = f"+{offset:6.2f}s  [{fmt_duration(duration):>8}]"
        fixed = len(task_str) + 2 + len(label) + 2 + len(timing)
        dots = "." * max(3, line_width - fixed)
        print(f"{task_str} {dots} {label}  {timing}")

        if not passed:
            exc_event = next(
                (e for e in t.get("events", []) if e.get("name") == "exception"),
                None,
            )
            if exc_event:
                attrs = exc_event.get("attributes", {})
                exc_type = attrs.get("exception.type", "")
                exc_msg = attrs.get("exception.message", "")
                detail = f"{exc_type}: {exc_msg}" if exc_type else exc_msg
            else:
                detail = t.get("status", {}).get("description", "")
            if detail:
                for ln in detail.splitlines()[:5]:
                    print(f"    {ln}")

    # Load resource samples once for use in all sections below
    rs = {}
    samples_path = Path(traces_path).parent / "resource_samples.json"
    if samples_path.exists():
        rs = json.loads(samples_path.read_text())

    last_end_ts = tasks[-1].get("end_time") or tasks[-1].get("start_time")
    wall = (parse_ts(last_end_ts) - workflow_start).total_seconds() if last_end_ts else total

    print()
    spec = print_machine_spec(traces_path, run_start=workflow_start)
    parallelism = total / wall if wall > 0 else 1.0

    # ── Timing section ──────────────────────────────────────────────────────
    time_rows = [
        ("time spent running tasks", f"{fmt_duration(total)}  ({total:.1f}s)"),
        ("total time start to finish", f"{fmt_duration(wall)}  ({wall:.1f}s)"),
        (
            "tasks ran at the same time",
            f"{parallelism:.2f}x  ({'yes—some tasks overlapped' if parallelism > 1.05 else 'no — one task at a time'})",
        ),
    ]
    avg_cpu = rs.get("avg_cpu_pct")
    if avg_cpu:
        time_rows.append(("average CPU usage", f"{avg_cpu}%"))
    peak_swap = rs.get("peak_swap_bytes", 0)
    if peak_swap:
        time_rows.append(
            ("memory swapped to disk", f"{fmt_bytes(peak_swap)}  (this slows things down — more free RAM would help)")
        )

    tw = max(len(k) for k, _ in time_rows)
    print()
    print("Timing:")
    for key, val in time_rows:
        print(f"  {key:<{tw}}  {val}")

    # ── Slowest tasks ────────────────────────────────────────────────────────
    def task_duration(t) -> float:
        return (parse_ts(t["end_time"]) - parse_ts(t["start_time"])).total_seconds()

    slowest = sorted(tasks, key=task_duration, reverse=True)[:5]
    print()
    print("Slowest tasks:")
    sw = max(len(t["name"]) for t in slowest)
    for t in slowest:
        print(f"  {t['name']:<{sw}}  {fmt_duration(task_duration(t))}")

    # ── Memory pressure warning ───────────────────────────────────────────────
    peak_rss = rs.get("peak_rss_bytes", 0)
    ram_available_gb = spec.get("ram_available_gb")
    if ram_available_gb and peak_rss:
        used_pct = peak_rss / (ram_available_gb * 1024**3) * 100
        if used_pct > 80:
            print()
            print(f"  Warning: the workflow used {used_pct:.0f}% of the memory that was free when it started")
            print(f"           ({fmt_bytes(peak_rss)} used out of {ram_available_gb} GB available).")
            print("           Running on a machine with more free memory will likely make it faster.")

    # ── Summary bar ──────────────────────────────────────────────────────────
    passed_count = len(tasks) - failed
    summary = f"{passed_count} passed"
    if failed:
        summary += f", {failed} failed"
    summary += f"  |  total {fmt_duration(total)}  wall {fmt_duration(wall)}"
    if rs.get("peak_rss_bytes"):
        summary += f"  |  peak memory {fmt_bytes(rs['peak_rss_bytes'])}"
    if rs.get("peak_cpu_pct"):
        summary += f"  |  peak CPU {rs['peak_cpu_pct']}%"
    disk_r = rs.get("disk_read_bytes")
    disk_w = rs.get("disk_write_bytes")
    if disk_r is not None and disk_w is not None:
        summary += f"  |  disk read {fmt_bytes(disk_r)}  written {fmt_bytes(disk_w)}"
    net_rx = rs.get("net_bytes_recv")
    net_tx = rs.get("net_bytes_sent")
    if net_rx is not None and net_tx is not None:
        summary += f"  |  network downloaded {fmt_bytes(net_rx)}  uploaded {fmt_bytes(net_tx)}"

    pad = max(2, (line_width - len(summary) - 2) // 2)
    bar = "=" * pad
    print()
    print(f"{bar} {summary} {bar}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parse-traces.py <otel_traces.jsonl>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
