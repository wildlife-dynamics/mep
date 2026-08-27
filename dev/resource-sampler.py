#!/usr/bin/env python3
"""Resource-monitoring wrapper: runs a command and writes resource_samples.json.

Usage: resource-sampler.py <results_dir> <command> [args...]
If results_dir is empty, the command is run without monitoring.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: resource-sampler.py <results_dir> <command> [args...]", file=sys.stderr)
        return 2

    results_dir = Path(sys.argv[1]) if sys.argv[1] else None
    argv = sys.argv[2:]

    peak_rss = 0
    peak_swap = 0
    cpu_samples = []
    stop_event = threading.Event()
    net_start = None
    io_start = None
    last_io = [None]
    thread = None

    proc = subprocess.Popen(argv)

    if results_dir:
        try:
            import psutil

            target = psutil.Process(proc.pid)
            target.cpu_percent(interval=None)  # prime — first call always returns 0.0

            try:
                net_start = psutil.net_io_counters()
            except Exception:
                pass
            try:
                io_start = target.io_counters()
                last_io[0] = io_start
            except Exception:
                pass

            def _sample():
                nonlocal peak_rss, peak_swap
                while not stop_event.wait(0.5):
                    try:
                        rss = target.memory_info().rss
                        cpu = target.cpu_percent(interval=None)
                        swap = psutil.swap_memory().used
                        if rss > peak_rss:
                            peak_rss = rss
                        if swap > peak_swap:
                            peak_swap = swap
                        if cpu > 0:
                            cpu_samples.append(cpu)
                        try:
                            last_io[0] = target.io_counters()
                        except Exception:
                            pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break

            thread = threading.Thread(target=_sample, daemon=True, name="resource-sampler")
            thread.start()

        except ImportError:
            pass

    ec = proc.wait()

    if results_dir and thread is not None:
        stop_event.set()
        thread.join(timeout=2)

        out = {
            "peak_rss_bytes": peak_rss,
            "peak_swap_bytes": peak_swap,
            "peak_cpu_pct": round(max(cpu_samples), 1) if cpu_samples else 0.0,
            "avg_cpu_pct": round(sum(cpu_samples) / len(cpu_samples), 1) if cpu_samples else 0.0,
            "sample_count": len(cpu_samples),
        }
        if net_start is not None:
            try:
                import psutil

                net_end = psutil.net_io_counters()
                out["net_bytes_recv"] = net_end.bytes_recv - net_start.bytes_recv
                out["net_bytes_sent"] = net_end.bytes_sent - net_start.bytes_sent
            except Exception:
                pass
        if io_start is not None and last_io[0] is not None:
            out["disk_read_bytes"] = last_io[0].read_bytes - io_start.read_bytes
            out["disk_write_bytes"] = last_io[0].write_bytes - io_start.write_bytes

        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "resource_samples.json").write_text(json.dumps(out, indent=2))

    return ec


if __name__ == "__main__":
    sys.exit(main())
