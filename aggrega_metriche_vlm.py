"""
Aggrega le metriche VLM da tutti i data.json giornalieri in 'diari/'
e stampa una tabella Markdown pronta da incollare nella tesi.

Uso:
    python aggrega_metriche_vlm.py [cartella_diari]
"""

import sys
import json
import glob
from pathlib import Path


LABELS = {
    "singolo": "Osservazione singola (1 frame)",
    "sequenza": "Osservazione di sequenza (≥2 frame)",
    "confronto": "Confronto ambientale",
    "sintesi_oraria": "Sintesi oraria",
    "diario_giornaliero": "Diario giornaliero",
    "diario_settimanale": "Diario settimanale",
    "diario_mensile": "Diario mensile",
    "telegram_classificazione": "Classificazione domanda Telegram",
    "telegram_risposta": "Risposta bot Telegram",
}

ORDER = list(LABELS.keys())


def aggregate(diari_dir):
    files = sorted(glob.glob(f"{diari_dir}/*/*/*/data.json"))
    totals_vlm = {}
    totals_rag = {}
    days_covered = set()

    for f in files:
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue

        vlm_metrics = data.get("vlm_metrics", {})
        rag_metrics = data.get("rag_metrics", {})
        if not vlm_metrics and not rag_metrics:
            continue
        days_covered.add(data.get("date") or Path(f).parent.name)

        for cat, stats in vlm_metrics.items():
            n = stats.get("n", 0)
            if n == 0:
                continue
            t = totals_vlm.setdefault(cat, {
                "n": 0, "ttft_sum": 0.0, "ttft_n": 0,
                "ttlt_sum": 0.0,
                "ttlt_min": float("inf"), "ttlt_max": float("-inf"),
            })
            t["n"] += n
            if stats.get("ttft_mean") is not None:
                t["ttft_sum"] += stats["ttft_mean"] * n
                t["ttft_n"] += n
            t["ttlt_sum"] += stats["ttlt_mean"] * n
            t["ttlt_min"] = min(t["ttlt_min"], stats.get("ttlt_min", t["ttlt_min"]))
            t["ttlt_max"] = max(t["ttlt_max"], stats.get("ttlt_max", t["ttlt_max"]))

        for phase, stats in rag_metrics.items():
            n = stats.get("n", 0)
            if n == 0:
                continue
            r = totals_rag.setdefault(phase, {
                "n": 0, "sum": 0.0,
                "min": float("inf"), "max": float("-inf"),
            })
            r["n"] += n
            r["sum"] += stats["mean"] * n
            r["min"] = min(r["min"], stats.get("min", r["min"]))
            r["max"] = max(r["max"], stats.get("max", r["max"]))

    return totals_vlm, totals_rag, sorted(days_covered)


RAG_LABELS = {
    "embedding": "Embedding della query (BGE-M3)",
    "retrieval": "Retrieval da ChromaDB",
    "build_context": "Costruzione del contesto",
    "total": "Totale fase di ricerca",
}
RAG_ORDER = ["embedding", "retrieval", "build_context", "total"]


def print_tables(totals_vlm, totals_rag, days):
    print(f"\nGiorni con metriche presenti: {len(days)}")
    if days:
        print(f"  Dal {days[0]} al {days[-1]}")

    if totals_vlm:
        print("\n## Tabella — Tempi di risposta del VLM per tipo di chiamata\n")
        print("| Tipo di chiamata | N | TTFT medio (s) | TTLT medio (s) | TTLT min (s) | TTLT max (s) |")
        print("|---|---:|---:|---:|---:|---:|")
        seen = set()
        for cat in ORDER + sorted(set(totals_vlm) - set(ORDER)):
            if cat in seen or cat not in totals_vlm:
                continue
            seen.add(cat)
            t = totals_vlm[cat]
            n = t["n"]
            ttft = round(t["ttft_sum"] / t["ttft_n"], 2) if t["ttft_n"] else "—"
            ttlt = round(t["ttlt_sum"] / n, 2)
            label = LABELS.get(cat, cat)
            print(f"| {label} | {n} | {ttft} | {ttlt} | {t['ttlt_min']:.2f} | {t['ttlt_max']:.2f} |")
        total_calls = sum(t["n"] for t in totals_vlm.values())
        print(f"\nChiamate VLM totali aggregate: {total_calls}")

    if totals_rag:
        print("\n## Tabella — Tempi della ricerca semantica (RAG)\n")
        print("| Fase | N | Tempo medio (s) | Min (s) | Max (s) |")
        print("|---|---:|---:|---:|---:|")
        seen = set()
        for phase in RAG_ORDER + sorted(set(totals_rag) - set(RAG_ORDER)):
            if phase in seen or phase not in totals_rag:
                continue
            seen.add(phase)
            r = totals_rag[phase]
            n = r["n"]
            mean = round(r["sum"] / n, 3)
            label = RAG_LABELS.get(phase, phase)
            print(f"| {label} | {n} | {mean} | {r['min']:.3f} | {r['max']:.3f} |")
        total_q = totals_rag.get("total", {}).get("n", 0)
        if total_q:
            print(f"\nRicerche RAG totali aggregate: {total_q}")


if __name__ == "__main__":
    diari_dir = sys.argv[1] if len(sys.argv) > 1 else "diari"
    if not Path(diari_dir).exists():
        print(f"Cartella non trovata: {diari_dir}")
        sys.exit(1)
    totals_vlm, totals_rag, days = aggregate(diari_dir)
    if not totals_vlm and not totals_rag:
        print("Nessuna metrica trovata nei data.json.")
        sys.exit(0)
    print_tables(totals_vlm, totals_rag, days)
