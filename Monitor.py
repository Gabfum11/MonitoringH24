"""
VLM Daily Monitor — Monitoraggio h24 con Vision Language Model.

Architettura a tre livelli:
  1. Cattura intelligente: confronta i frame e chiama il VLM solo quando
     la scena cambia o è passato troppo tempo dall'ultima osservazione.
  2. Sintesi oraria: ogni ora condensa le osservazioni in un paragrafo.
  3. Diario giornaliero: a fine giornata, sintetizza i riepiloghi orari
     in un diario narrativo di 2-3 pagine.

Uso:
    python vlm_monitor.py
    python vlm_monitor.py --interval 30 --diary-hour 22
"""

import cv2
import time
import mss
import json
import base64
import argparse
import requests
import numpy as np
from datetime import datetime, date
from pathlib import Path


class VLMMonitor:
    def __init__(self,
                 model="gemma-4-26b-a4b-it", # Modello VLM da LM Studio
                 lmstudio_url="http://localhost:1234", # URL del server locale  LM Studio
                 capture_interval=30, # Intervallo base in secondi tra le catture (adattivo)
                 monitor_area=None, #area dello schermo da catturare
                 diary_hour=24, #quando generare il diario
                 output_dir="diari"): #dove salvare i file . lo salva in "diari/YYYY-MM-DD_data.json" e "diari/YYYY-MM-DD_diario.txt"
        self.model = model
        self.lmstudio_url = lmstudio_url
        self.capture_interval = capture_interval
        self.diary_hour = diary_hour
        self.output_dir = Path(output_dir) #crea la cartella se non esiste
        self.output_dir.mkdir(exist_ok=True) #evita errori se la cartella esiste già

        # Screen capture
        self.sct = mss.mss()
        if monitor_area:
            self.monitor = monitor_area  
        else:
            self.monitor = {"top": 270, "left": 10, "width": 900, "height": 520} #area dello schermo da catturare

        # Stato giornata
        self.today = date.today().isoformat() # "YYYY-MM-DD"
        self.diary_generated = False #serve per evitare di generare più di un diario se il programma viene riavviato durante la stessa giornata

        # Osservazioni grezze (livello 1)
        self.observations = [] #ora, descrizione, tipo (singolo o sequenza), timestamp ISO 

        # Riepiloghi orari (livello 2)
        self.hourly_summaries = [] #ora, numero osservazioni, testo del riepilogo sintetico

        # Change detection
        self._prev_frame_gray = None #frame precendente 
        self._prev_observation_time = 0 #timestamp dell'ultima osservazione
        self._same_scene_count = 0

        # Intervalli adattivi
        self._min_interval = capture_interval       # 30s durante attività (minimo tempo tra osservazioni)
        self._max_interval = 900                    # 15 min se scena stabile (notte)
        self._current_interval = capture_interval
        self._no_change_streak = 0

        # Tracking orario
        self._last_hourly_summary = datetime.now().hour #per sapere quando generare il riepilogo orario successivo

        # Prompt di sistema
        self.system_prompt = (
        "Sei un assistente per il monitoraggio domiciliare di una persona anziana.\n\n"
        "Analizza l'immagine e descrivi SOLO informazioni rilevanti dal punto di vista clinico.\n\n"
        "In ogni risposta, valuta sempre:\n"
        "- Presenza o assenza della persona\n"
        "- Postura (seduta, in piedi, sdraiata)\n"
        "- Attività in corso\n"
        "- Stabilità e movimento (normale, lento, incerto)\n"
        "- Possibili segnali di rischio (caduta, immobilità prolungata, difficoltà)\n\n"
        "Se nelle osservazioni precedenti la persona era in una posizione diversa, "
        "segnala il cambiamento.\n\n"
        "Scrivi in italiano, massimo 2-3 frasi, stile cartella clinica.\n"
        "Sii oggettivo, preciso e sintetico.\n"
        "NON descrivere dettagli irrilevanti (arredamento, luce, ecc.) "
        "a meno che non siano importanti per la sicurezza.\n"
        "Se la persona non è visibile, dichiaralo chiaramente."
    )
        # Carica dati esistenti se il programma viene riavviato
        self._load_existing_data()  

    # =========================================
    # CATTURA E CHANGE DETECTION
    # =========================================
    def _capture_frame(self):
        """Cattura un frame dallo schermo."""
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _frame_to_base64(self, frame):
        """Ridimensiona e converte un frame in JPEG base64."""
        h, w = frame.shape[:2]
        max_size = 512
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(jpg.tobytes()).decode('utf-8')

    def _scene_changed(self, frame):
        """Confronta il frame corrente con il precedente.
        
        Differenza media assoluta su versione a bassa risoluzione.
        Soglia 5: sotto è rumore/luce, sopra è movimento reale.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #elimina i colori per semplificare il confronto
        gray = cv2.resize(gray, (160, 120)) #ridimensiona per velocizzare il confronto

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return True

        diff = np.mean(np.abs(gray.astype(float) - self._prev_frame_gray.astype(float))) #differenza media assoluta tra i pixel del frame corrente e quello precedente
        self._prev_frame_gray = gray

        return diff > 5 #se è minotre di 5, consideriamo che la scena è stabile, altrimenti è cambiata (movimento reale)
        #è giusta la soglia di 5?dipende molto dalla scena, dalla luce, dal monitor. In generale è un buon punto di partenza per distinguere tra piccoli cambiamenti (rumore, luce) e movimenti reali. Se si notano troppe osservazioni durante la notte, si può aumentare a 7-10. Se invece si perdono movimenti importanti, si può abbassare a 3-4.
    def _capture_burst(self, n_frames=4, interval=0.5):
        """Cattura una sequenza rapida di frame per analizzare un'azione.
        
        Usato quando la scena cambia: il VLM vede il movimento,
        non solo una posa statica.
        
        Args:
            n_frames: quanti frame catturare (3-5)
            interval: secondi tra i frame (0.5 = 2fps)
        """
        frames = []
        for i in range(n_frames):
            frame = self._capture_frame()
            frames.append(self._frame_to_base64(frame))
            if i < n_frames - 1:
                time.sleep(interval)
        return frames

    # ==================-=======================
    # INTERVALLO ADATTIVO
    # =========================================
    def _update_interval(self, scene_changed):
        """Adatta l'intervallo: attività → 30s, stabile/notte → fino a 15 min.
        permette di evitare lo spreco di risorse quando non succede nulla
        se la scena è attiva la controlla spesso 
        """
        if scene_changed:
            self._no_change_streak = 0 
            self._current_interval = self._min_interval
        else:
            self._no_change_streak += 1
            if self._no_change_streak % 3 == 0:
                self._current_interval = min(
                    self._current_interval * 2,
                    self._max_interval
                )

    # =========================================
    # DECISIONE: CHIAMARE IL VLM?
    # =========================================
    def _should_observe(self, scene_changed): 
        """Decide se e come osservare.
        scene_changed viene calcolato confrontando il frame corrente con quello precedente. 
        Se è cambiato, significa che c'è un movimento o un'attività in corso, 
        quindi è più probabile che ci siano informazioni rilevanti da catturare. 
        Se invece la scena è stabile, 
    potrebbe essere inutile chiamare il VLM troppo spesso, 
    soprattutto durante la notte quando la persona potrebbe essere immobile, quindi un frame singolo ogni tanto può essere sufficiente 
        
        Ritorna:
            None: non osservare
            'single': frame singolo (scena stabile, check periodico)
            'burst': sequenza di frame (scena che cambia, azione in corso)
        """
        now = time.time()
        time_since_last = now - self._prev_observation_time

        # Scena cambiata → burst per catturare l'azione
        if scene_changed and time_since_last >= 15:
            return 'burst'

        # Troppo tempo senza osservazioni → frame singolo di controllo
        if time_since_last >= self._max_interval:
            return 'single'

        return None

    # =========================================
    # CHIAMATA VLM
    # =========================================
    def _call_vlm(self, images_b64, context_messages=None, max_tokens=200, prompt_text=None):
        """Invia una o più immagini a LM Studio con contesto.
        
        Args:
            images_b64: stringa base64 (singola) o lista di stringhe (sequenza)
            context_messages: contesto conversazionale
            max_tokens: token massimi
            prompt_text: testo personalizzato (default: "Descrivi cosa vedi")
        """
        messages = [{"role": "system", "content": self.system_prompt}] 

        if context_messages:
            messages.extend(context_messages)

        now = datetime.now().strftime("%H:%M:%S")
        
        # Costruisci il contenuto con una o più immagini
        if isinstance(images_b64, list):
            # Sequenza di frame — il VLM vede un'azione
            if prompt_text is None:
                prompt_text = (
                    f"Ore {now}. Questa è una sequenza di {len(images_b64)} frame consecutivi "
                    f"catturati in pochi secondi. Descrivi l'azione in corso: "
                    f"cosa sta facendo la persona? Come si muove? "
                    f"Noti difficoltà, instabilità o qualcosa di rilevante?"
                )
            content = [{"type": "text", "text": prompt_text}]
            for img in images_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                })
        else:
            # Frame singolo — scena statica
            if prompt_text is None:
                prompt_text = f"Ore {now}. Descrivi cosa vedi."
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_b64}"}}
            ]

        messages.append({"role": "user", "content": content})

        try:
            response = requests.post(
                f"{self.lmstudio_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[VLM] Errore: {response.status_code}")
                return None

        except Exception as e:
            print(f"[VLM] Errore connessione: {e}")
            return None

    def _call_vlm_text(self, prompt, system=None, max_tokens=800):
        """Chiamata solo testo (per sintesi e diario)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.lmstudio_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.4
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"[VLM] Errore: {response.status_code}")
                return None

        except Exception as e:
            print(f"[VLM] Errore: {e}")
            return None

    # =========================================
    # CONTESTO
    # =========================================
    def _build_context(self):
        """Ultime 3 osservazioni come contesto conversazionale."""
        if not self.observations:
            return None

        recent = self.observations[-3:]
        context = []
        for obs in recent:
            context.append({
                "role": "user",
                "content": f"Ore {obs['time']}. Descrivi cosa vedi."
            })
            context.append({
                "role": "assistant",
                "content": obs['description']
            })
        return context

    # =========================================
    # LIVELLO 1: OSSERVAZIONE
    # =========================================
    def _observe(self, frame, mode='single'):
        """Analizza il frame (o una sequenza) e salva l'osservazione.
        
        Args:
            frame: frame corrente (usato per 'single')
            mode: 'single' per frame singolo, 'burst' per sequenza rapida
        """
        context = self._build_context()

        if mode == 'burst':
            # Cattura sequenza rapida per vedere l'azione
            images = self._capture_burst(n_frames=4, interval=0.5)
            description = self._call_vlm(images, context, max_tokens=250)
            obs_type = "sequenza"
        else:
            # Frame singolo per scena statica
            image_b64 = self._frame_to_base64(frame)
            description = self._call_vlm(image_b64, context)
            obs_type = "singolo"

        if description:
            obs = {
                "time": datetime.now().strftime("%H:%M"),
                "timestamp": datetime.now().isoformat(),
                "hour": datetime.now().hour,
                "type": obs_type,
                "description": description
            }
            self.observations.append(obs)
            self._prev_observation_time = time.time()
            self._save_data()
            tag = "SEQ" if mode == 'burst' else "   "
            print(f"[{obs['time']}] [{tag}] {description}")
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] Nessuna risposta dal VLM")
            return False

    # =========================================
    # LIVELLO 2: SINTESI ORARIA
    # =========================================
    def _generate_hourly_summary(self, hour):
        """Genera un riepilogo per un'ora specifica."""
        hour_obs = [o for o in self.observations if o.get('hour') == hour]

        if not hour_obs:
            return

        # Evita duplicati
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

        summary = self._call_vlm_text(
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
            self._save_data()
            print(f"\n[SINTESI {hour:02d}:00] {summary}\n")

    def _check_hourly_summary(self):
        """Controlla se è ora di generare una sintesi oraria."""
        current_hour = datetime.now().hour
        if current_hour != self._last_hourly_summary:
            prev_hour = self._last_hourly_summary
            self._generate_hourly_summary(prev_hour)
            self._last_hourly_summary = current_hour

    # =========================================
    # LIVELLO 3: DIARIO GIORNALIERO
    # =========================================
    def generate_diary(self):
        """Genera il diario giornaliero dai riepiloghi orari."""
        # Genera l'ultimo riepilogo orario
        current_hour = datetime.now().hour
        self._generate_hourly_summary(current_hour)

        if not self.hourly_summaries and not self.observations:
            print("[DIARIO] Nessun dato, diario non generato")
            return None

        print(f"\n{'='*60}")
        print(f"[DIARIO] Generazione diario giornaliero...")
        print(f"  Osservazioni: {len(self.observations)}")
        print(f"  Riepiloghi orari: {len(self.hourly_summaries)}")
        print(f"{'='*60}")

        # Usa i riepiloghi orari se disponibili
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

        first_time = self.observations[0]['time'] if self.observations else "N/D"
        last_time = self.observations[-1]['time'] if self.observations else "N/D"

        prompt = (
            f"Oggi {self.today}, il sistema ha monitorato la persona dalle {first_time} alle {last_time}.\n"
            f"Totale osservazioni: {len(self.observations)}.\n\n"
            f"Ecco i {source} della giornata:\n\n"
            f"{content}\n\n"
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

        diary = self._call_vlm_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare. Scrivi in modo dettagliato e professionale.",
            max_tokens=3000
        )

        if diary:
            self._save_diary(diary)
            return diary
        else:
            print("[DIARIO] Errore nella generazione")
            return None

    def _save_diary(self, diary_text):
        first_time = self.observations[0]['time'] if self.observations else "N/D"
        last_time = self.observations[-1]['time'] if self.observations else "N/D"

        header = (
            f"DIARIO DI MONITORAGGIO DOMICILIARE\n"
            f"{'='*50}\n"
            f"Data: {self.today}\n"
            f"Periodo: {first_time} - {last_time}\n"
            f"Osservazioni: {len(self.observations)}\n"
            f"Riepiloghi orari: {len(self.hourly_summaries)}\n"
            f"{'='*50}\n\n"
        )

        path = self.output_dir / f"diario_{self.today}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + diary_text)
        print(f"[DIARIO] Salvato in {path}")

    # =========================================
    # PERSISTENZA
    # =========================================
    def _data_path(self):
        return self.output_dir / f"data_{self.today}.json"

    def _save_data(self):
        data = {
            "date": self.today,
            "observations": self.observations,
            "hourly_summaries": self.hourly_summaries
        }
        with open(self._data_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_existing_data(self):
        path = self._data_path()
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.observations = data.get("observations", [])
            self.hourly_summaries = data.get("hourly_summaries", [])
            print(f"[INIT] Caricati {len(self.observations)} osservazioni e "
                  f"{len(self.hourly_summaries)} riepiloghi per {self.today}")

    # =========================================
    # CAMBIO GIORNATA
    # =========================================
    def _check_new_day(self):
        today = date.today().isoformat()
        if today != self.today:
            if not self.diary_generated and self.observations:
                print(f"[DIARIO] Cambio giornata, genero diario per {self.today}")
                self.generate_diary()

            self.today = today
            self.observations = []
            self.hourly_summaries = []
            self.diary_generated = False
            self._last_hourly_summary = datetime.now().hour
            self._load_existing_data()
            print(f"[INIT] Nuovo giorno: {self.today}")

    # =========================================
    # LOOP PRINCIPALE
    # =========================================
    def run(self):
        print(f"{'='*60}")
        print(f"VLM Daily Monitor — Monitoraggio h24")
        print(f"  Modello:      {self.model}")
        print(f"  Server:       {self.lmstudio_url}")
        print(f"  Intervallo:   {self.capture_interval}s (adattivo)")
        print(f"  Diario:       ore {self.diary_hour}:00")
        print(f"  Output:       {self.output_dir}")
        print(f"{'='*60}")
        print("Premi Ctrl+C per fermare e generare il diario\n")

        try:
            while True:
                self._check_new_day()
                self._check_hourly_summary()

                frame = self._capture_frame()
                changed = self._scene_changed(frame)
                self._update_interval(changed)

                obs_mode = self._should_observe(changed)
                if obs_mode:
                    self._observe(frame, mode=obs_mode)

                now = datetime.now()
                if now.hour >= self.diary_hour and not self.diary_generated:
                    self.generate_diary()
                    self.diary_generated = True

                time.sleep(self._current_interval)

        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("[STOP] Interruzione manuale")
            if self.observations and not self.diary_generated:
                print("[DIARIO] Generazione diario finale...")
                self.generate_diary()
            print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="VLM Daily Monitor h24")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    parser.add_argument("--url", default="http://localhost:1234")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--diary-hour", type=int, default=22)
    parser.add_argument("--output", default="diari")
    parser.add_argument("--top", type=int, default=130)
    parser.add_argument("--left", type=int, default=100)
    parser.add_argument("--width", type=int, default=870)
    parser.add_argument("--height", type=int, default=520)

    args = parser.parse_args()

    monitor = VLMMonitor(
        model=args.model,
        lmstudio_url=args.url,
        capture_interval=args.interval,
        monitor_area={
            "top": args.top, "left": args.left,
            "width": args.width, "height": args.height
        },
        diary_hour=args.diary_hour,
        output_dir=args.output
    )
    monitor.run()


if __name__ == "__main__":
    main()