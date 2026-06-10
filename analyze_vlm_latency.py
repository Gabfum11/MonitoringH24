"""
Analizza monitor.log e calcola statistiche di latenza del VLM (TTFT/TTLT)
aggregate per tipo di chiamata.

Uso:
    python analyze_vlm_latency.py [path/to/monitor.log]
"""

import re
import sys
import statistics
from collections import defaultdict
from pathlib import Path


VLM_RE = re.compile(r"\[VLM\]\s+TTFT=([\d.]+|None)s\s+TTLT=([\d.]+)s")

OBS_TAGS = {
    "FIX": "Osservazione singola (1 frame)",
    "EVT": "Osservazione di sequenza (4 frame)",
    "EVT_R": "Osservazione di sequenza rapida",
    "CMP": "Confronto ambientale",
}
OBS_LINE_RE = re.compile(r"^\[\d{2}:\d{2}\]\s+\[(FIX|EVT|EVT_R|CMP)(?:×(\d+))?\]")

SINTESI_RE = re.compile(r"^\[SINTESI\s+\d{2}:00\]")
DIARIO_TAG_RE = re.compile(r"^\[DIARIO\]\s+Generazione")
TELEGRAM_CTX_RE = re.compile(r"\[TELEGRAM\]")


def classify(prev_context, next_line):
    """Determina il tipo di chiamata VLM in base al contesto e alla riga successiva."""
    if next_line:
        m = OBS_LINE_RE.match(next_line)
        if m:
            return OBS_TAGS[m.group(1)]
        if SINTESI_RE.match(next_line):
            return "Sintesi oraria"
    if prev_context == "diario":
        return "Generazione diario giornaliero"
    if prev_context == "telegram":
        return "Risposta bot Telegram"
    return "Altro / non classificato"


def analyze(log_path):
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()

    samples = defaultdict(lambda: {"ttft": [], "ttlt": []})

    prev_context = None
    for i, line in enumerate(lines):
        if DIARIO_TAG_RE.search(line):
            prev_context = "diario"
            continue
        if TELEGRAM_CTX_RE.search(line) and "Bot avviato" not in line:
            prev_context = "telegram"
            continue

        m = VLM_RE.search(line)
        if not m:
            continue

        ttft_raw, ttlt_raw = m.group(1), m.group(2)
        ttft = None if ttft_raw == "None" else float(ttft_raw)
        ttlt = float(ttlt_raw)

        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        category = classify(prev_context, next_line)

        if ttft is not None:
            samples[category]["ttft"].append(ttft)
        samples[category]["ttlt"].append(ttlt)

        prev_context = None

    return samples


def fmt(values):
    if not values:
        return "—"
    return f"{statistics.mean(values):.2f}"


def fmt_range(values):
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{min(values):.2f} – {max(values):.2f}"


def print_report(samples):
    order = [
        "Osservazione singola (1 frame)",
        "Osservazione di sequenza (4 frame)",
        "Osservazione di sequenza rapida",
        "Confronto ambientale",
        "Sintesi oraria",
        "Generazione diario giornaliero",
        "Risposta bot Telegram",
        "Altro / non classificato",
    ]

    print("\n## Tabella — Tempi di risposta del VLM per tipo di chiamata\n")
    print("| Tipo di chiamata | N | TTFT medio (s) | TTLT medio (s) | TTLT range (s) |")
    print("|---|---:|---:|---:|---:|")
    for cat in order:
        if cat not in samples:
            continue
        d = samples[cat]
        n = len(d["ttlt"])
        if n == 0:
            continue
        print(f"| {cat} | {n} | {fmt(d['ttft'])} | {fmt(d['ttlt'])} | {fmt_range(d['ttlt'])} |")

    total = sum(len(d["ttlt"]) for d in samples.values())
    print(f"\nCampioni totali: {total}")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "monitor.log"
    if not Path(log).exists():
        print(f"File non trovato: {log}")
        sys.exit(1)
    samples = analyze(log)
    print_report(samples)
