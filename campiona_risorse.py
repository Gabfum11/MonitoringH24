"""
Campionamento continuo delle risorse hardware del sistema.

Lancialo in parallelo al monitoraggio (Monitor.py) e a eventuali test/chiamate VLM.
Al termine stampa media, picchi e timestamp dei picchi per ogni risorsa.

Uso:
    python campiona_risorse.py [minuti]   # default: 30 min

Misura: CPU%, RAM%, RAM (GB), Disk I/O (KB/s), GPU/Energy (se sudo disponibile).

Per le metriche GPU/energia su Apple Silicon serve sudo:
    sudo python campiona_risorse.py 30
"""

import os
import sys
import csv
import time
import shutil
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import psutil


SAMPLE_INTERVAL = 2.0   # secondi tra campioni


def sample_psutil(prev_disk):
    """Restituisce un dict con CPU%, RAM%, RAM GB, disk KB/s."""
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    if prev_disk is None:
        kbs = 0.0
    else:
        delta = (disk.read_bytes + disk.write_bytes) - (prev_disk.read_bytes + prev_disk.write_bytes)
        kbs = delta / SAMPLE_INTERVAL / 1024
    return {
        "cpu_percent": cpu,
        "ram_percent": mem.percent,
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "disk_kbps": round(kbs, 1),
    }, disk


def start_powermetrics(out_path):
    """Avvia powermetrics in background, scrive su file. Richiede sudo."""
    if os.geteuid() != 0:
        return None
    cmd = [
        "powermetrics",
        "--samplers", "cpu_power,gpu_power",
        "-i", str(int(SAMPLE_INTERVAL * 1000)),
        "-o", str(out_path),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_powermetrics(path):
    """Estrae CPU Power (mW), GPU Power (mW), GPU active residency (%) dal file."""
    if not path or not Path(path).exists():
        return [], [], []
    cpu_w, gpu_w, gpu_act = [], [], []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("CPU Power:"):
                try:
                    cpu_w.append(float(line.split(":")[1].split()[0]) / 1000)
                except Exception:
                    pass
            elif line.startswith("GPU Power:"):
                try:
                    gpu_w.append(float(line.split(":")[1].split()[0]) / 1000)
                except Exception:
                    pass
            elif "GPU HW active residency" in line:
                try:
                    gpu_act.append(float(line.split(":")[1].split("%")[0].strip()))
                except Exception:
                    pass
    return cpu_w, gpu_w, gpu_act


def fmt(values, fn, suffix="", digits=2):
    if not values:
        return "—"
    return f"{fn(values):.{digits}f}{suffix}"


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    duration = minutes * 60

    log_path = Path(f"campioni_risorse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pm_path = Path(f"powermetrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    is_root = os.geteuid() == 0

    print(f"[CAMPIONAMENTO] Durata: {minutes:.1f} min  |  Output CSV: {log_path}")
    if is_root:
        print(f"[CAMPIONAMENTO] powermetrics attivo (richiede sudo) → {pm_path}")
    else:
        print("[CAMPIONAMENTO] sudo NON attivo → GPU/energia non disponibili.")
        print("                Per misurarle: sudo python campiona_risorse.py")

    pm_proc = start_powermetrics(pm_path) if is_root else None

    samples = []
    prev_disk = None
    psutil.cpu_percent(interval=None)  # warm-up
    t_start = time.time()
    next_tick = t_start

    try:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_percent", "ram_percent", "ram_used_gb", "disk_kbps"])

            while time.time() - t_start < duration:
                next_tick += SAMPLE_INTERVAL
                row, prev_disk = sample_psutil(prev_disk)
                ts = datetime.now().isoformat(timespec="seconds")
                writer.writerow([ts, row["cpu_percent"], row["ram_percent"], row["ram_used_gb"], row["disk_kbps"]])
                samples.append(row)
                sleep_for = max(0, next_tick - time.time())
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[CAMPIONAMENTO] Interrotto manualmente.")
    finally:
        if pm_proc:
            pm_proc.terminate()
            try:
                pm_proc.wait(timeout=5)
            except Exception:
                pm_proc.kill()

    if not samples:
        print("Nessun campione raccolto.")
        return

    cpu = [s["cpu_percent"] for s in samples]
    ram_pct = [s["ram_percent"] for s in samples]
    ram_gb = [s["ram_used_gb"] for s in samples]
    disk = [s["disk_kbps"] for s in samples]

    cpu_w, gpu_w, gpu_act = parse_powermetrics(pm_path) if is_root else ([], [], [])

    print(f"\n=== RIEPILOGO ({len(samples)} campioni in {minutes:.1f} min) ===\n")
    print(f"{'Risorsa':<28} {'Media':>10} {'Picco':>10}")
    print("-" * 50)
    print(f"{'CPU di sistema (%)':<28} {fmt(cpu, statistics.mean, '%', 1):>10} {fmt(cpu, max, '%', 1):>10}")
    print(f"{'Memoria usata (GB)':<28} {fmt(ram_gb, statistics.mean, ' GB', 1):>10} {fmt(ram_gb, max, ' GB', 1):>10}")
    print(f"{'Memoria usata (%)':<28} {fmt(ram_pct, statistics.mean, '%', 1):>10} {fmt(ram_pct, max, '%', 1):>10}")
    print(f"{'I/O disco (KB/s)':<28} {fmt(disk, statistics.mean, ' KB/s', 1):>10} {fmt(disk, max, ' KB/s', 1):>10}")
    if cpu_w:
        print(f"{'Potenza CPU (W)':<28} {fmt(cpu_w, statistics.mean, ' W', 2):>10} {fmt(cpu_w, max, ' W', 2):>10}")
    if gpu_w:
        print(f"{'Potenza GPU (W)':<28} {fmt(gpu_w, statistics.mean, ' W', 2):>10} {fmt(gpu_w, max, ' W', 2):>10}")
    if gpu_act:
        print(f"{'GPU active (%)':<28} {fmt(gpu_act, statistics.mean, '%', 1):>10} {fmt(gpu_act, max, '%', 1):>10}")
    total_w = []
    if cpu_w and gpu_w:
        total_w = [c + g for c, g in zip(cpu_w, gpu_w)]
        print(f"{'Consumo totale (CPU+GPU)':<28} {fmt(total_w, statistics.mean, ' W', 2):>10} {fmt(total_w, max, ' W', 2):>10}")

    print(f"\nCSV raw salvato in: {log_path}")
    if is_root:
        print(f"Log powermetrics in: {pm_path}")


if __name__ == "__main__":
    if not shutil.which("powermetrics") and os.geteuid() == 0:
        print("Attenzione: 'powermetrics' non trovato nel PATH.")
    main()
