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


class DiaryGenerator:
    def __init__(self, vlm_client, observations, hourly_summaries, output_dir="diari"):
        """
        Args:
            vlm_client: istanza di VLMClient
            observations: lista condivisa delle osservazioni
            hourly_summaries: lista condivisa dei riepiloghi orari
            output_dir: cartella base per i file
        """
        self.vlm = vlm_client
        self.observations = observations
        self.hourly_summaries = hourly_summaries
        self.output_dir = Path(output_dir)
        self.today = date.today().isoformat()

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

    def _get_annual_dir(self, year):
        """Es: diari/2026/"""
        path = self.output_dir / str(year)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================
    # PERSISTENZA DATI GIORNALIERI
    # =========================================
    def data_path(self):
        """Path del file data.json per oggi."""
        return self._get_daily_dir() / "data.json"

    def save_data(self):
        """Salva osservazioni e riepiloghi orari su disco."""
        data = {
            "date": self.today,
            "observations": self.observations,
            "hourly_summaries": self.hourly_summaries
        }
        with open(self.data_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
            print(f"[INIT] Caricati {len(self.observations)} osservazioni e "
                  f"{len(self.hourly_summaries)} riepiloghi per {self.today}")

    # =========================================
    # LIVELLO 1: SINTESI ORARIA
    # =========================================
    def generate_hourly_summary(self, hour):
        """Genera un riepilogo per un'ora specifica."""
        hour_obs = [o for o in self.observations if o.get('hour') == hour]
        if not hour_obs:
            return

        existing_hours = {s['hour'] for s in self.hourly_summaries}
        if hour in existing_hours:
            return

        obs_text = "\n".join(f"- {o['time']}: {o['description']}" for o in hour_obs)

        prompt = (
            f"Ecco le osservazioni raccolte tra le {hour}:00 e le {hour}:59:\n\n"
            f"{obs_text}\n\n"
            f"Scrivi un paragrafo di 3-5 frasi che riassuma questo periodo. "
            f"Cosa ha fatto la persona? Come stava? Ci sono state variazioni? "
            f"Eventuali segnali di attenzione?\n"
            f"Scrivi in italiano, in modo professionale e conciso."
        )

        summary = self.vlm.call_text(
            prompt,
            system="Sei un assistente clinico per il monitoraggio domiciliare.",
            max_tokens=300
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
            print(f"\n[SINTESI {hour:02d}:00] {summary}\n")

    # =========================================
    # LIVELLO 2: DIARIO GIORNALIERO
    # =========================================
    def generate_diary(self):
        """Genera il diario giornaliero dai riepiloghi orari."""
        from datetime import datetime
        current_hour = datetime.now().hour
        self.generate_hourly_summary(current_hour)

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

        alerts = [o for o in self.observations if o.get('type') == 'alert']
        alert_text = ""
        if alerts:
            alert_text = (
                f"\n\nATTENZIONE — Durante la giornata sono stati generati {len(alerts)} alert:\n"
                + "\n".join(f"- {a['time']}: {a['description']}" for a in alerts)
            )

        first_time = self.observations[0]['time'] if self.observations else "N/D"
        last_time = self.observations[-1]['time'] if self.observations else "N/D"

        prompt = (
            f"Oggi {self.today}, il sistema ha monitorato la persona dalle {first_time} alle {last_time}.\n"
            f"Totale osservazioni: {len(self.observations)}.\n\n"
            f"Ecco i {source} della giornata:\n\n"
            f"{content}{alert_text}\n\n"
            f"Scrivi un DIARIO GIORNALIERO completo in italiano (2-3 pagine). Struttura:\n\n"
            f"RIEPILOGO GENERALE: 3-4 frasi sullo stato complessivo della persona.\n\n"
            f"MATTINA (6:00-12:00): cosa ha fatto, come stava, eventuali difficoltà.\n\n"
            f"POMERIGGIO (12:00-18:00): attività, riposo, cambiamenti.\n\n"
            f"SERA/NOTTE (18:00-6:00): cena, preparazione al sonno, qualità del riposo.\n\n"
            f"PATTERN E SEGNALAZIONI: periodi di inattività prolungata, "
            f"difficoltà motorie ricorrenti, cambiamenti rispetto ai giorni precedenti.\n\n"
            f"VALUTAZIONE CLINICA: impressione complessiva e raccomandazioni.\n\n"
            f"Scrivi in modo professionale ma comprensibile per un medico o un caregiver."
        )

        diary = self.vlm.call_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare. Scrivi in modo dettagliato e professionale.",
            max_tokens=3000
        )

        if diary:
            self._save_diary(diary, first_time, last_time)
            return diary
        else:
            print("[DIARIO] Errore nella generazione")
            return None

    def _save_diary(self, diary_text, first_time, last_time):
        types = {}
        for o in self.observations:
            t = o.get('type', 'singolo')
            types[t] = types.get(t, 0) + 1

        header = (
            f"DIARIO DI MONITORAGGIO DOMICILIARE\n"
            f"{'='*50}\n"
            f"Data: {self.today}\n"
            f"Periodo: {first_time} - {last_time}\n"
            f"Osservazioni totali: {len(self.observations)}\n"
            f"  Singole: {types.get('singolo', 0)}\n"
            f"  Sequenze: {types.get('sequenza', 0) + types.get('sequenza_rapida', 0)}\n"
            f"  Confronti: {types.get('confronto', 0)}\n"
            f"  Alert: {types.get('alert', 0)}\n"
            f"Riepiloghi orari: {len(self.hourly_summaries)}\n"
            f"{'='*50}\n\n"
        )

        path = self._get_daily_dir() / "diario.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + diary_text)
        print(f"[DIARIO] Salvato in {path}")

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

        content = "\n\n".join(
            f"--- {entry['date']} ---\n{entry['content']}"
            for entry in daily_diaries
        )

        prompt = (
            f"Sei un geriatra. Ecco i diari giornalieri di monitoraggio domiciliare "
            f"dal {start_date.isoformat()} al {end_date.isoformat()} "
            f"({len(daily_diaries)} giorni su 7 con dati disponibili).\n\n"
            f"{content}\n\n"
            f"Scrivi un REPORT SETTIMANALE completo in italiano (2-3 pagine). Struttura:\n\n"
            f"RIEPILOGO SETTIMANALE: stato complessivo della persona durante la settimana.\n\n"
            f"ANDAMENTO GIORNALIERO: per ogni giorno, una sintesi di 2-3 frasi.\n\n"
            f"PATTERN SETTIMANALI: pattern ricorrenti, orari di maggiore attività, "
            f"momenti di difficoltà, evoluzione della mobilità.\n\n"
            f"CONFRONTO E TREND: miglioramento, peggioramento o stabilità?\n\n"
            f"RACCOMANDAZIONI CLINICHE: suggerimenti basati sui pattern osservati.\n\n"
            f"Scrivi in modo professionale, per un medico di base o un caregiver."
        )

        diary = self.vlm.call_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare a lungo termine.",
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
        daily_diaries = self._read_daily_diaries(start_date, end_date)

        if not weekly_reports and not daily_diaries:
            print("[MENSILE] Nessun dato trovato, skip")
            return None

        if weekly_reports:
            # Settimanali per il quadro generale
            content = "REPORT SETTIMANALI:\n\n"
            content += "\n\n".join(
                f"--- Settimana {entry['period']} ---\n{entry['content']}"
                for entry in weekly_reports
            )
            # Aggiungi anche i giornalieri per il dettaglio
            if daily_diaries:
                content += "\n\nDETTAGLIO GIORNALIERO:\n\n"
                content += "\n\n".join(
                    f"--- {entry['date']} ---\n{entry['content']}"
                    for entry in daily_diaries
                )
            source = (f"{len(weekly_reports)} report settimanali + "
                      f"{len(daily_diaries)} diari giornalieri")
        else:
            # Fallback: solo giornalieri — tronca solo se necessario
            # Stima token: ~1.4 token per parola
            total_words = sum(len(e['content'].split()) for e in daily_diaries)
            max_words = 35000  # ~50000 token lasciando spazio per prompt e risposta
            
            if total_words > max_words:
                # Tronca proporzionalmente ogni diario
                words_per_diary = max_words // len(daily_diaries)
                truncated = []
                for entry in daily_diaries:
                    words = entry['content'].split()
                    short = ' '.join(words[:words_per_diary])
                    if len(words) > words_per_diary:
                        short += "\n[...troncato...]"
                    truncated.append({"date": entry['date'], "content": short})
                daily_diaries = truncated
                source = f"{len(daily_diaries)} diari giornalieri (troncati)"
            else:
                source = f"{len(daily_diaries)} diari giornalieri"

            content = "\n\n".join(
                f"--- {entry['date']} ---\n{entry['content']}"
                for entry in daily_diaries
            )

        prompt = (
            f"Sei un geriatra. Ecco i dati di monitoraggio domiciliare per {month_name} "
            f"(fonte: {source}).\n\n"
            f"{content}\n\n"
            f"Scrivi un REPORT MENSILE completo in italiano (3-4 pagine). Struttura:\n\n"
            f"RIEPILOGO DEL MESE: stato complessivo della persona.\n\n"
            f"ANDAMENTO SETTIMANALE: per ogni settimana, 3-4 frasi.\n\n"
            f"EVOLUZIONE DELLA MOBILITÀ: autonomia rispetto all'inizio del mese?\n\n"
            f"PATTERN MENSILI: orari ricorrenti, giorni migliori/peggiori, eventi critici.\n\n"
            f"CONFRONTO CON IL MESE PRECEDENTE: miglioramenti o peggioramenti.\n\n"
            f"VALUTAZIONE CLINICA E RACCOMANDAZIONI: impressione e suggerimenti.\n\n"
            f"Scrivi in modo professionale, per un medico o un geriatra."
        )

        diary = self.vlm.call_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare a lungo termine.",
            max_tokens=5000
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
            return diary
        else:
            print("[MENSILE] Errore nella generazione")
            return None

    # =========================================
    # LIVELLO 5: REPORT ANNUALE
    # =========================================
    def generate_annual_diary(self, year=None):
        """Genera il report annuale dai report mensili."""
        if year is None:
            year = date.today().year - 1

        print(f"\n{'='*60}")
        print(f"[ANNUALE] Generazione diario annuale: {year}")
        print(f"{'='*60}")

        # Cerca i report mensili — li usa completi (12 × ~2500 token = ~30000, dentro i 64K)
        monthly_reports = []
        for month in range(1, 13):
            month_dir = self._get_monthly_dir(year, month)
            path = month_dir / f"mensile_{year}-{month:02d}.txt"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                body = content.split('\n\n', 1)
                body = body[1] if len(body) > 1 else content
                monthly_reports.append({
                    "month": f"{year}-{month:02d}",
                    "content": body.strip()
                })

        if not monthly_reports:
            print("[ANNUALE] Nessun report mensile trovato, skip")
            return None

        print(f"  Mesi con dati: {len(monthly_reports)}/12")

        content = "\n\n".join(
            f"--- {entry['month']} ---\n{entry['content']}"
            for entry in monthly_reports
        )

        prompt = (
            f"Sei un geriatra. Ecco i report mensili di monitoraggio domiciliare "
            f"per l'anno {year} ({len(monthly_reports)} mesi su 12 con dati).\n\n"
            f"{content}\n\n"
            f"Scrivi un REPORT ANNUALE completo in italiano (4-5 pagine). Struttura:\n\n"
            f"RIEPILOGO ANNUALE: stato complessivo della persona durante l'anno.\n\n"
            f"EVOLUZIONE TRIMESTRALE: per ogni trimestre, 4-5 frasi sull'andamento.\n\n"
            f"MOBILITÀ E AUTONOMIA: come è cambiata nel corso dell'anno.\n\n"
            f"PATTERN STAGIONALI: differenze tra estate e inverno, periodi migliori e peggiori.\n\n"
            f"EVENTI SIGNIFICATIVI: cadute, ospedalizzazioni, cambiamenti improvvisi.\n\n"
            f"VALUTAZIONE CLINICA COMPLESSIVA: impressione generale, "
            f"prognosi, suggerimenti per il piano di cura annuale.\n\n"
            f"Scrivi in modo professionale, per un geriatra o un medico di base."
        )

        diary = self.vlm.call_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare a lungo termine.",
            max_tokens=6000
        )

        if diary:
            header = (
                f"REPORT ANNUALE DI MONITORAGGIO DOMICILIARE\n"
                f"{'='*50}\n"
                f"Anno: {year}\n"
                f"Mesi con dati: {len(monthly_reports)}/12\n"
                f"{'='*50}\n\n"
            )
            annual_dir = self._get_annual_dir(year)
            path = annual_dir / f"annuale_{year}.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(header + diary)
            print(f"[ANNUALE] Salvato in {path}")
            return diary
        else:
            print("[ANNUALE] Errore nella generazione")
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