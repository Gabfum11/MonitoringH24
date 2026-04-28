"""
VLM Daily Monitor — Monitoraggio h24 con Vision Language Model.

Modulo principale (orchestratore). Compone i moduli:
  - capture.py: cattura frame, change detection, burst
  - vlm_client.py: chiamate al VLM (LM Studio)
  - observer.py: osservazione, ridondanza, intervalli, assenza, confronto
  - diary_generator.py: sintesi orarie, diari giornalieri/settimanali/mensili/annuali

Uso:
    python Monitor.py                  # monitoraggio h24
    python Monitor.py --preview        # anteprima area catturata
    python Monitor.py --gen-weekly     # genera report settimanale
    python Monitor.py --gen-monthly    # genera report mensile
    python Monitor.py --gen-annual     # genera report annuale
"""

import time
import argparse
from datetime import datetime, date, timedelta

from Capture import CaptureManager
from Vlm_calls import VLMClient
from Observer import Observer
from Diary_generator import DiaryGenerator
import psutil


class VLMMonitor:
    def __init__(self,
                 model="gemma-4-26b-a4b-it",
                 lmstudio_url="http://localhost:1234",
                 capture_interval=30,
                 monitor_area=None,
                 output_dir="diari"):

        # Dati condivisi (passati per riferimento a tutti i moduli)
        self.observations = []
        self.hourly_summaries = []

        # Stato giornata
        self.today = date.today().isoformat()
        self.diary_generated = False

        # Inizializza i moduli
        self.capture = CaptureManager(
            monitor_area=monitor_area,
            use_window_capture=False
        )
        self.vlm = VLMClient(model=model, lmstudio_url=lmstudio_url)
        self.diary = DiaryGenerator(
            self.vlm, self.observations, self.hourly_summaries,
            output_dir=output_dir
        )
        self.observer = Observer(
            self.capture, self.vlm, self.observations,
            save_callback=self.diary.save_data,
            capture_interval=capture_interval
        )

        # Tracking orario
        self._last_hourly_summary = datetime.now().hour

        # Carica dati esistenti
        self.diary.load_existing_data()

    # =========================================
    # CAMBIO GIORNATA
    # =========================================
    def _check_new_day(self):
        today = date.today()
        today_str = today.isoformat()
        if today_str != self.today:
            # Genera la sintesi dell'ultima ora prima del diario
            self.diary.generate_hourly_summary(self._last_hourly_summary)

            # Diario giornaliero
            if not self.diary_generated and self.observations:
                print(f"[DIARIO] Cambio giornata, genero diario per {self.today}")
                self.diary.generate_diary()

            # Lunedì → report settimanale
            if today.weekday() == 0:
                yesterday = today - timedelta(days=1)
                print(f"[SETTIMANALE] È lunedì, genero report settimanale")
                self.diary.generate_weekly_diary(end_date=yesterday)

            # 1° del mese → report mensile
            if today.day == 1:
                print(f"[MENSILE] Primo del mese, genero report mensile")
                self.diary.generate_monthly_diary()

            # 1° gennaio → report annuale
            if today.month == 1 and today.day == 1:
                print(f"[ANNUALE] Primo gennaio, genero report annuale")
                self.diary.generate_annual_diary()

            # Reset per il nuovo giorno
            self.today = today_str
            self.diary.today = today_str
            self.observations.clear()
            self.hourly_summaries.clear()
            self.diary_generated = False
            self._last_hourly_summary = datetime.now().hour
            self.observer.reset()
            self.diary.load_existing_data()
            print(f"[INIT] Nuovo giorno: {self.today}")

    def _check_hourly_summary(self):
        """Controlla se è ora di generare una sintesi oraria."""
        current_hour = datetime.now().hour
        if current_hour != self._last_hourly_summary:
            self.diary.generate_hourly_summary(self._last_hourly_summary)
            self._last_hourly_summary = current_hour

    # =========================================
    # LOOP PRINCIPALE
    # =========================================
    def run(self):
        print(f"{'='*60}")
        print(f"VLM Daily Monitor — Monitoraggio h24")
        print(f"  Modello:      {self.vlm.model}")
        print(f"  Server:       {self.vlm.lmstudio_url}")
        print(f"  Cattura:      {self.capture.capture_mode}")
        print(f"  Intervallo:   {self.observer._min_interval}s (adattivo)")
        print(f"  Diario:       a mezzanotte (automatico)")
        print(f"  Output:       {self.diary.output_dir}")
        print(f"{'='*60}")
        print("Premi Ctrl+C per fermare e generare il diario\n")

        try:
            while True:
                self._check_new_day()
                self._check_hourly_summary()

                # Cattura e change detection
                frame = self.capture.capture_frame()
                changed = self.capture.scene_changed(frame)

                # Aggiorna intervallo
                self.observer.update_interval(changed, self.capture.last_diff)

                # Osservazione (include assenza tracking internamente)
                obs_mode = self.observer.should_observe(changed, self.capture.last_diff)
                if obs_mode:
                    self.observer.observe(frame, mode=obs_mode)

                # Confronto ambientale ogni ora
                self.observer.check_comparison(frame)

                time.sleep(10)
                # Monitor CPU (debug)
                cpu_percent = psutil.cpu_percent(interval=0)
                if cpu_percent > 50:  # stampa solo se alto
                    print(f"[CPU] {cpu_percent}%")

        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("[STOP] Interruzione manuale")
            if self.observations and not self.diary_generated:
                print("[DIARIO] Generazione diario finale...")
                self.diary.generate_diary()
            print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="VLM Daily Monitor h24")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    parser.add_argument("--url", default="http://localhost:1234")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--output", default="diari")
    parser.add_argument("--top", type=int, default=270)
    parser.add_argument("--left", type=int, default=10)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--preview", action="store_true",
                        help="Mostra anteprima dell'area catturata")
    parser.add_argument("--gen-weekly", action="store_true",
                        help="Genera il report settimanale e esci")
    parser.add_argument("--gen-monthly", action="store_true",
                        help="Genera il report mensile e esci")
    parser.add_argument("--gen-annual", action="store_true",
                        help="Genera il report annuale e esci")

    args = parser.parse_args()

    monitor = VLMMonitor(
        model=args.model,
        lmstudio_url=args.url,
        capture_interval=args.interval,
        monitor_area={
            "top": args.top, "left": args.left,
            "width": args.width, "height": args.height
        },
        output_dir=args.output
    )

    if args.gen_weekly:
        monitor.diary.generate_weekly_diary()
        return
    if args.gen_monthly:
        monitor.diary.generate_monthly_diary()
        return
    if args.gen_annual:
        monitor.diary.generate_annual_diary()
        return
    if args.preview:
        monitor.capture.preview()

    monitor.run()


if __name__ == "__main__":
    main()