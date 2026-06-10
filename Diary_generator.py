"""
Generatore di diari clinici.

Gestisce la gerarchia di sintesi a 5 livelli:
  1. Sintesi oraria → paragrafo per ogni ora
  2. Diario giornaliero → 2-3 pagine dai riepiloghi orari
  3. Report settimanale → 2-3 pagine dai 7 diari giornalieri
  4. Report mensile → 3-4 pagine dai report settimanali
  5. Report annuale → 4-5 pagine dai report mensili

Gestisce anche la persistenza (salvataggio/caricamento osservazioni)
e la struttura delle cartelle:
  diari/
  └── 2026/
      ├── annuale_2026.txt
      └── 04/
          ├── mensile_2026-04.txt
          ├── settimanale_2026-04-14_2026-04-20.txt
          └── 2026-04-24/
              ├── diario.txt
              └── data.json
"""

import json
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from Database_manager import DatabaseManager


class DiaryGenerator:
    def __init__(self, vlm_client, observations, hourly_summaries, output_dir="diari", db=None, rag=None):
        """
        Args:
            vlm_client: istanza di VLMClient
            observations: lista condivisa delle osservazioni
            hourly_summaries: lista condivisa dei riepiloghi orari
            output_dir: cartella base per i file
            db: istanza di DatabaseManager (opzionale, ne crea una se None)
            rag: istanza di RagIndex (opzionale)
        """
        self.vlm = vlm_client
        self.observations = observations
        self.hourly_summaries = hourly_summaries
        self.output_dir = Path(output_dir)
        self.today = date.today().isoformat()
        self.db = db or DatabaseManager()
        self.rag = rag
        # Lock per scritture/letture concorrenti su data.json
        # (Monitor.py e TestRunner possono salvare in thread diversi).
        self._save_lock = Lock()

    # =========================================
    # GESTIONE CARTELLE
    # =========================================
    def _get_daily_dir(self, day_str=None):
        """Es: diari/2026/04/2026-04-24/"""
        if day_str is None:
            day_str = self.today
        d = date.fromisoformat(day_str)
        path = self.output_dir / str(d.year) / f"{d.month:02d}" / day_str
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_monthly_dir(self, year, month):
        """Es: diari/2026/04/"""
        path = self.output_dir / str(year) / f"{month:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

# =========================================
    # TEST CLINICI
    # =========================================
    def _format_test_data(self, start_date_str, end_date_str):
        """Costruisce un blocco testo con i test clinici nel periodo, pronto per il prompt."""
        tug = self.db.get_tug_results(start_date_str, end_date_str)
        sts = self.db.get_sts_results(start_date_str, end_date_str)
        if not tug and not sts:
            return ""

        lines = [f"\nTEST CLINICI ({start_date_str} → {end_date_str}):"]

        if tug:
            lines.append("\nTUG (Timed Up and Go):")
            tug = [r for r in tug if r['total_time'] is not None]
            for r in tug:
                lines.append(f"  {r['date']}: {r['total_time']:.1f}s")
            if len(tug) >= 2:
                delta_t = tug[-1]['total_time'] - tug[0]['total_time']
                if delta_t > 1:
                    lines.append(f"  → andamento: peggioramento (+{delta_t:.1f}s rispetto al primo test)")
                elif delta_t < -1:
                    lines.append(f"  → andamento: miglioramento ({abs(delta_t):.1f}s rispetto al primo test)")
                else:
                    lines.append("  → andamento: stabile")

        if sts:
            lines.append("\nSTS (5 Sit-to-Stand):")
            sts = [r for r in sts if r['total_time'] is not None]
            for r in sts:
                base = f"  {r['date']}: {r['total_time']:.1f}s | {r['reps_completed']} rip."
                rep_times = r.get('rep_times') or []
                if rep_times:
                    reps_str = ", ".join(f"rep{rt['numero']}={rt['tempo']:.1f}s" for rt in rep_times)
                    base += f" | {reps_str}"
                lines.append(base)
            if len(sts) >= 2:
                delta = sts[-1]['total_time'] - sts[0]['total_time']
                if delta > 1:
                    lines.append(f"  peggioramento (+{delta:.1f}s rispetto al primo test)")
                elif delta < -1:
                    lines.append(f"  miglioramento ({abs(delta):.1f}s rispetto al primo test)")
                else:
                    lines.append(" stabile")

        return "\n".join(lines) + "\n"

    # =========================================
    # PERSISTENZA DATI GIORNALIERI
    # =========================================
    def data_path(self):
        """Path del file data.json per oggi."""
        return self._get_daily_dir() / "data.json"

    def save_data(self):
        """Salva osservazioni, riepiloghi orari e metriche di latenza su disco.

        La scrittura è atomica (write su file .tmp + rename) e protetta da lock
        per evitare race condition tra Monitor.py e TestRunner.
        """
        with self._save_lock:
            data = {
                "date": self.today,
                "observations": self.observations,
                "hourly_summaries": self.hourly_summaries,
                "vlm_metrics": self._merged_vlm_metrics(),
                "rag_metrics": self._merged_rag_metrics()
            }
            target = self.data_path()
            tmp = target.with_suffix(target.suffix + '.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(target)

    def _merged_rag_metrics(self):
        """Combina le metriche RAG su disco con quelle accumulate in sessione."""
        existing = {}
        path = self.data_path()
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing = json.load(f).get("rag_metrics", {}) or {}
            except Exception:
                existing = {}

        current = self.rag.get_latency_summary() if (self.rag and hasattr(self.rag, "get_latency_summary")) else {}
        merged = dict(existing)
        for phase, stats in current.items():
            prev = merged.get(phase)
            if not prev:
                merged[phase] = stats
                continue
            n_prev, n_new = prev["n"], stats["n"]
            n_tot = n_prev + n_new
            merged[phase] = {
                "n": n_tot,
                "mean": round((prev["mean"] * n_prev + stats["mean"] * n_new) / n_tot, 3),
                "min": round(min(prev["min"], stats["min"]), 3),
                "max": round(max(prev["max"], stats["max"]), 3),
            }
        if self.rag and hasattr(self.rag, "reset_latency"):
            self.rag.reset_latency()
        return merged

    def _merged_vlm_metrics(self):
        """Combina le metriche già su disco con quelle accumulate in sessione,
        cosicché un riavvio dello stesso giorno non azzeri il conteggio."""
        existing = {}
        path = self.data_path()
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing = json.load(f).get("vlm_metrics", {}) or {}
            except Exception:
                existing = {}

        current = self.vlm.get_latency_summary() if hasattr(self.vlm, "get_latency_summary") else {}
        merged = dict(existing)
        for cat, stats in current.items():
            prev = merged.get(cat)
            if not prev:
                merged[cat] = stats
                continue
            n_prev, n_new = prev["n"], stats["n"]
            n_tot = n_prev + n_new

            def weighted(key):
                a, b = prev.get(key), stats.get(key)
                if a is None and b is None:
                    return None
                if a is None:
                    return b
                if b is None:
                    return a
                return round((a * n_prev + b * n_new) / n_tot, 2)

            merged[cat] = {
                "n": n_tot,
                "ttft_mean": weighted("ttft_mean"),
                "ttlt_mean": weighted("ttlt_mean"),
                "ttlt_min": round(min(prev.get("ttlt_min", stats["ttlt_min"]), stats["ttlt_min"]), 2),
                "ttlt_max": round(max(prev.get("ttlt_max", stats["ttlt_max"]), stats["ttlt_max"]), 2),
            }
        if hasattr(self.vlm, "reset_latency"):
            self.vlm.reset_latency()
        return merged

    def load_existing_data(self):
        """Carica dati esistenti se il programma viene riavviato."""
        path = self.data_path()
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.observations.clear()
            self.observations.extend(data.get("observations", []))
            self.hourly_summaries.clear()
            self.hourly_summaries.extend(data.get("hourly_summaries", []))
            print(f"Caricati {len(self.observations)} osservazioni e "
                  f"{len(self.hourly_summaries)} riepiloghi per {self.today}")

    # =========================================
    # LIVELLO 1: SINTESI ORARIA
    # =========================================
    def generate_hourly_summary(self, hour, allow_silent=False):
        """Genera un riepilogo per un'ora specifica.

        Se allow_silent=True e l'ora non ha osservazioni, viene comunque creato
        un placeholder per garantire la copertura completa delle 24 ore.
        """
        existing_hours = {s['hour'] for s in self.hourly_summaries}
        if hour in existing_hours:
            return

        hour_obs = [o for o in self.observations if o.get('hour') == hour]
        if not hour_obs:
            if not allow_silent:
                return
            entry = {
                "hour": hour,
                "hour_label": f"{hour:02d}:00-{hour:02d}:59",
                "n_observations": 0,
                "summary": "Nessuna osservazione registrata in questa fascia oraria."
            }
            self.hourly_summaries.append(entry)
            self.save_data()
            if self.rag:
                self.rag.index_summary(self.today, hour, entry["hour_label"], entry["summary"])
            print(f"[SINTESI {hour:02d}:00] (ora silente)")
            return

        obs_text = "\n".join(f"- {o['time']}: {o['description']}" for o in hour_obs)

        prompt = (
            f"Ecco le osservazioni raccolte tra le {hour}:00 e le {hour}:59:\n\n"
            f"{obs_text}\n\n"
            f"Scrivi un paragrafo di 3-5 frasi che riassuma questo periodo. "
            f"Cosa ha fatto la persona? Come stava? Ci sono state variazioni? "
            f"Eventuali segnali di attenzione?\n"
            f"Scrivi in italiano, in modo professionale e conciso, in testo semplice senza formattazione Markdown."
        )

        summary = self.vlm.call_text(
            prompt,
            system="Sei un assistente clinico per il monitoraggio domiciliare.",
            max_tokens=300,
            category="sintesi_oraria"
        )

        if summary:
            entry = {
                "hour": hour,
                "hour_label": f"{hour:02d}:00-{hour:02d}:59",
                "n_observations": len(hour_obs),
                "summary": summary
            }
            self.hourly_summaries.append(entry)
            self.save_data()
            if self.rag:
                self.rag.index_summary(self.today, hour, entry["hour_label"], summary)
            print(f"\n[SINTESI {hour:02d}:00] {summary}\n")

    def _backfill_hourly_summaries(self):
        """Garantisce che tutte le ore passate della giornata abbiano un riepilogo.

        - Se il diario è per una giornata già chiusa, copre tutte le 24 ore.
        - Se è per la giornata corrente (chiamata manuale mid-day), copre solo
          le ore passate, non quella in corso o future.
        """
        from datetime import datetime
        today_str = date.today().isoformat()
        max_hour_exclusive = datetime.now().hour if self.today == today_str else 24

        existing_hours = {s['hour'] for s in self.hourly_summaries}
        missing = [h for h in range(max_hour_exclusive) if h not in existing_hours]
        if not missing:
            return
        print(f"[BACKFILL] Genero {len(missing)} riepiloghi orari mancanti: {missing}")
        for h in missing:
            self.generate_hourly_summary(h, allow_silent=True)

    # =========================================
    # LIVELLO 2: DIARIO GIORNALIERO
    # =========================================
    def generate_diary(self):
        """Genera il diario giornaliero dai riepiloghi orari."""
        from datetime import datetime
        current_hour = datetime.now().hour
        self.generate_hourly_summary(current_hour)
        self._backfill_hourly_summaries()

        if not self.hourly_summaries and not self.observations:
            print("[DIARIO] Nessun dato, diario non generato")
            return None

        print(f"\n{'='*60}")
        print(f"[DIARIO] Generazione diario giornaliero...")
        print(f"  Osservazioni: {len(self.observations)}")
        print(f"  Riepiloghi orari: {len(self.hourly_summaries)}")
        print(f"{'='*60}")

        if self.hourly_summaries:
            content = "\n\n".join(
                f"[{s['hour_label']}] ({s['n_observations']} osservazioni)\n{s['summary']}"
                for s in sorted(self.hourly_summaries, key=lambda x: x['hour'])
            )
            source = "riepiloghi orari"
        else:
            content = "\n".join(
                f"- Ore {o['time']}: {o['description']}"
                for o in self.observations
            )
            source = "osservazioni grezze"

        assenze = [o for o in self.observations if o.get('type') == 'assenza']
        alert_text = ""
        if assenze:
            alert_text = (
                f"\n\nATTENZIONE — Durante la giornata sono state registrate {len(assenze)} assenze prolungate:\n"
                + "\n".join(f"- {a['time']}: {a['description']}" for a in assenze)
            )

        first_time = self.observations[0]['time'] if self.observations else "N/D"
        last_time = self.observations[-1]['time'] if self.observations else "N/D"

        observed_hours = {o.get('hour') for o in self.observations}
        if observed_hours:
            first_hour = min(observed_hours)
            last_hour = max(observed_hours)
            missing_hours = [h for h in range(first_hour, last_hour + 1) if h not in observed_hours]
            missing_hours_note = ""
            if missing_hours:
                labels = ", ".join(f"{h:02d}:00-{h:02d}:59" for h in missing_hours)
                missing_hours_note = f"\nATTENZIONE: nessuna osservazione nelle seguenti ore: {labels}.\n"
        else:
            missing_hours_note = ""

        week_ago = (date.fromisoformat(self.today) - timedelta(days=7)).isoformat()
        test_block = self._format_test_data(week_ago, self.today)

        prompt = (
            f"Oggi {self.today}, il sistema ha monitorato la persona dalle {first_time} alle {last_time}.\n"
            f"Totale osservazioni: {len(self.observations)}.\n"
            f"{missing_hours_note}\n"
            f"Ecco i {source} della giornata:\n\n"
            f"{content}{alert_text}"
            f"{test_block}\n\n"
            f"Scrivi un DIARIO GIORNALIERO completo in italiano (2-3 pagine). Struttura:\n\n"
            f"RIEPILOGO GENERALE: 3-4 frasi sullo stato complessivo della persona.\n\n"
            f"NOTTE E PRIMA MATTINA (0:00-6:00): qualità del riposo, eventuali risvegli o spostamenti.\n\n"
            f"MATTINA (6:00-12:00): cosa ha fatto, come stava, eventuali difficoltà.\n\n"
            f"POMERIGGIO (12:00-18:00): attività, riposo, cambiamenti.\n\n"
            f"SERA (18:00-24:00): cena, attività serali, preparazione al sonno.\n\n"
            f"Se ci sono test clinici (TUG, STS) o eventi particolari, descrivili in una sezione dedicata.\n\n"
            f"PATTERN E SEGNALAZIONI: periodi di inattività prolungata, "
            f"difficoltà motorie ricorrenti, cambiamenti rispetto ai giorni precedenti.\n\n"
            "OSSERVAZIONI RILEVANTI: eventuali elementi che meritano attenzione "
            "(posture anomale, difficoltà nei movimenti, periodi di inattività prolungata, "
            "assenze dall'inquadratura).\n\n"
            "NON aggiungere firme, intestazioni fittizie, nomi di medici o formule di chiusura.\n\n"
            "Scrivi in testo semplice, senza formattazione Markdown.\n\n"
            "Scrivi in modo professionale ma comprensibile per un medico o un caregiver."
        )

        diary = self.vlm.call_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare. Scrivi in modo dettagliato e professionale.",
            max_tokens=3000,
            category="diario_giornaliero"
        )

        if diary:
            self._save_diary(diary, first_time, last_time)
            return diary
        else:
            print("[DIARIO] Errore nella generazione")
            return None

    def _save_diary(self, diary_text, first_time, last_time):
        header = (
            f"DIARIO DI MONITORAGGIO DOMICILIARE\n"
            f"{'='*50}\n"
            f"Data: {self.today}\n"
            f"Periodo: {first_time} - {last_time}\n"
            f"{'='*50}\n\n"
        )

        path = self._get_daily_dir() / "diario.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + diary_text)
        print(f"[DIARIO] Salvato in {path}")
        if self.rag:
            self.rag.index_diary(self.today, diary_text)
            print(f"[RAG] Diario giornaliero indicizzato")

    # =========================================
    # LIVELLO 3: REPORT SETTIMANALE
    # =========================================
    def generate_weekly_diary(self, end_date=None):
        """Genera il report settimanale dagli ultimi 7 diari giornalieri."""
        if end_date is None:
            end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=6)

        print(f"\n{'='*60}")
        print(f"[SETTIMANALE] Generazione diario settimanale")
        print(f"  Periodo: {start_date.isoformat()} → {end_date.isoformat()}")
        print(f"{'='*60}")

        daily_diaries = self._read_daily_diaries(start_date, end_date)
        if not daily_diaries:
            print("[SETTIMANALE] Nessun diario giornaliero trovato, skip")
            return None

        available_dates = {entry['date'] for entry in daily_diaries}
        missing_dates = [
            (start_date + timedelta(days=i)).isoformat()
            for i in range(7)
            if (start_date + timedelta(days=i)).isoformat() not in available_dates
        ]
        missing_note = ""
        if missing_dates:
            missing_note = f"ATTENZIONE: dati mancanti per i seguenti giorni: {', '.join(missing_dates)}.\n\n"

        content = "\n\n".join(
            f"--- {entry['date']} ---\n{entry['content']}"
            for entry in daily_diaries
        )

        test_block = self._format_test_data(start_date.isoformat(), end_date.isoformat())

        prompt = (
            f"Sei un geriatra. Ecco i diari giornalieri di monitoraggio domiciliare "
            f"dal {start_date.isoformat()} al {end_date.isoformat()} "
            f"({len(daily_diaries)} giorni su 7 con dati disponibili).\n\n"
            f"{missing_note}"
            f"{content}"
            f"{test_block}\n\n"
            f"Scrivi un REPORT SETTIMANALE completo in italiano (2-3 pagine). Struttura:\n\n"
            f"RIEPILOGO SETTIMANALE: stato complessivo della persona durante la settimana.\n\n"
            f"ANDAMENTO GIORNALIERO: per ogni giorno, una sintesi di 2-3 frasi.\n\n"
            f"PATTERN SETTIMANALI: pattern ricorrenti, orari di maggiore attività, "
            f"momenti di difficoltà, evoluzione della mobilità.\n\n"
            f"CONFRONTO E TREND: miglioramento, peggioramento o stabilità? "
            f"Se ci sono test clinici (TUG, STS), commentane i valori e il andamento.\n\n"
            "ELEMENTI DI ATTENZIONE: variazioni rispetto ai giorni precedenti, "
            "eventi ricorrenti, cambiamenti nel livello di attività o autonomia.\n\n"
            "Non aggiungere firme, intestazioni mediche o formule di chiusura.\n\n"
            "Scrivi in testo semplice, senza formattazione Markdown."
        )

        diary = self.vlm.call_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare a lungo termine.",
            category="diario_settimanale",
            max_tokens=4000
        )

        if diary:
            header = (
                f"REPORT SETTIMANALE DI MONITORAGGIO DOMICILIARE\n"
                f"{'='*50}\n"
                f"Periodo: {start_date.isoformat()} → {end_date.isoformat()}\n"
                f"Giorni con dati: {len(daily_diaries)}/7\n"
                f"{'='*50}\n\n"
            )
            month_dir = self._get_monthly_dir(end_date.year, end_date.month)
            path = month_dir / f"settimanale_{start_date.isoformat()}_{end_date.isoformat()}.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(header + diary)
            print(f"[SETTIMANALE] Salvato in {path}")
            if self.rag:
                self.rag.index_weekly(start_date.isoformat(), end_date.isoformat(), diary)
                print(f"[RAG] Report settimanale indicizzato")
            return diary
        else:
            print("[SETTIMANALE] Errore nella generazione")
            return None

    # =========================================
    # LIVELLO 4: REPORT MENSILE
    # =========================================
    def generate_monthly_diary(self, year=None, month=None):
        """Genera il report mensile dai report settimanali o diari giornalieri."""
        if year is None or month is None:
            today = date.today()
            if today.month == 1:
                year = today.year - 1
                month = 12
            else:
                year = today.year
                month = today.month - 1

        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        month_name = start_date.strftime("%B %Y")

        print(f"\n{'='*60}")
        print(f"[MENSILE] Generazione diario mensile: {month_name}")
        print(f"  Periodo: {start_date.isoformat()} → {end_date.isoformat()}")
        print(f"{'='*60}")

        weekly_reports = self._read_weekly_diaries(start_date, end_date)

        if not weekly_reports:
            print("[MENSILE] Nessun report settimanale trovato, genero con dati mancanti segnalati")

        content = "\n\n".join(
            f"--- Settimana {entry['period']} ---\n{entry['content']}"
            for entry in weekly_reports
        ) if weekly_reports else ""
        source = f"{len(weekly_reports)} report settimanali" if weekly_reports else "nessun dato disponibile"

        available_weeks = {entry['period'].split(' → ')[0] for entry in weekly_reports}
        missing_weeks = []
        cursor = start_date
        while cursor <= end_date:
            week_monday = cursor - timedelta(days=cursor.weekday())
            if week_monday.isoformat() not in available_weeks:
                week_sunday = week_monday + timedelta(days=6)
                missing_weeks.append(f"{week_monday.isoformat()} → {week_sunday.isoformat()}")
            cursor += timedelta(weeks=1)
            cursor -= timedelta(days=cursor.weekday())
        missing_weeks = list(dict.fromkeys(missing_weeks))
        missing_weeks_note = ""
        if missing_weeks:
            missing_weeks_note = f"ATTENZIONE: dati mancanti per le seguenti settimane: {'; '.join(missing_weeks)}.\n\n"

        test_block = self._format_test_data(start_date.isoformat(), end_date.isoformat())

        prompt = (
            f"Sei un geriatra. Ecco i dati di monitoraggio domiciliare per {month_name} "
            f"(fonte: {source}).\n\n"
            f"{missing_weeks_note}"
            f"{content}"
            f"{test_block}\n\n"
            f"Scrivi un REPORT MENSILE completo in italiano (3-4 pagine). Struttura:\n\n"
            f"RIEPILOGO DEL MESE: stato complessivo della persona.\n\n"
            f"ANDAMENTO SETTIMANALE: per ogni settimana, 3-4 frasi.\n\n"
            f"EVOLUZIONE DELLA MOBILITÀ: autonomia rispetto all'inizio del mese?\n\n"
            f"PATTERN MENSILI: orari ricorrenti, giorni migliori/peggiori, eventi critici.\n\n"
            f"TEST CLINICI: se presenti, commenta i valori TUG/STS, il andamento nel mese "
            f"e il confronto con il mese precedente.\n\n"
            f"CONFRONTO CON IL MESE PRECEDENTE: miglioramenti o peggioramenti.\n\n"
            "SINTESI DELLE VARIAZIONI: cambiamenti osservati nel periodo, "
            "andamento nell'attività e nella mobilità, eventi significativi registrati.\n\n"
            "Scrivi in testo semplice, senza formattazione Markdown."
        )

        diary = self.vlm.call_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare a lungo termine.",
            max_tokens=5000,
            category="diario_mensile"
        )

        if diary:
            header = (
                f"REPORT MENSILE DI MONITORAGGIO DOMICILIARE\n"
                f"{'='*50}\n"
                f"Mese: {month_name}\n"
                f"Periodo: {start_date.isoformat()} → {end_date.isoformat()}\n"
                f"{'='*50}\n\n"
            )
            month_dir = self._get_monthly_dir(year, month)
            path = month_dir / f"mensile_{year}-{month:02d}.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(header + diary)
            print(f"[MENSILE] Salvato in {path}")
            if self.rag:
                self.rag.index_monthly(year, month, diary)
                print(f"[RAG] Report mensile indicizzato")
            return diary
        else:
            print("[MENSILE] Errore nella generazione")
            return None

    # =========================================
    # LETTURA DIARI PRECEDENTI
    # =========================================
    def _read_daily_diaries(self, start_date, end_date):
        """Legge i diari giornalieri nel range di date."""
        diaries = []
        current = start_date
        while current <= end_date:
            day_dir = (self.output_dir / str(current.year) /
                       f"{current.month:02d}" / current.isoformat())
            path = day_dir / "diario.txt"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                parts = content.split('\n\n', 1)
                body = parts[1] if len(parts) > 1 else content
                diaries.append({
                    "date": current.isoformat(),
                    "content": body.strip()
                })
            current += timedelta(days=1)
        print(f"[LETTURA] Trovati {len(diaries)} diari giornalieri "
              f"su {(end_date - start_date).days + 1} giorni")
        return diaries

    def _read_weekly_diaries(self, start_date, end_date):
        """Legge i report settimanali nel range di date."""
        reports = []
        current_month = date(start_date.year, start_date.month, 1)
        while current_month <= end_date:
            month_dir = (self.output_dir / str(current_month.year) /
                         f"{current_month.month:02d}")
            if month_dir.exists():
                for path in sorted(month_dir.glob("settimanale_*.txt")):
                    try:
                        parts = path.stem.replace("settimanale_", "").split("_")
                        week_start = date.fromisoformat(parts[0])
                        week_end = date.fromisoformat(parts[1])
                        if week_start <= end_date and week_end >= start_date:
                            with open(path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            body = content.split('\n\n', 1)
                            body = body[1] if len(body) > 1 else content
                            reports.append({
                                "period": f"{week_start.isoformat()} → {week_end.isoformat()}",
                                "content": body.strip()
                            })
                    except (ValueError, IndexError):
                        continue
            if current_month.month == 12:
                current_month = date(current_month.year + 1, 1, 1)
            else:
                current_month = date(current_month.year, current_month.month + 1, 1)
        print(f"[LETTURA] Trovati {len(reports)} report settimanali nel periodo")
        return reports