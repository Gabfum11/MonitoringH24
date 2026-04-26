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
    python vlm_monitor.py --preview
    python vlm_monitor.py --interval 30
"""

import cv2
import time
import mss
import json
import base64
import argparse
import requests
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import deque

# Cattura finestra specifica su macOS (anche se coperta da altre finestre)
try:
    import Quartz
    from Quartz import (CGWindowListCopyWindowInfo, kCGWindowListOptionAll,
                        kCGNullWindowID, CGWindowListCreateImage,
                        kCGWindowImageDefault, CGRectNull,
                        kCGWindowListOptionIncludingWindow)
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False


class VLMMonitor:
    def __init__(self,
                 model="gemma-4-26b-a4b-it",       # Modello VLM da LM Studio
                 lmstudio_url="http://localhost:1234", # URL del server locale LM Studio
                 capture_interval=30,                # Intervallo base in secondi (adattivo)
                 monitor_area=None,                  # Area dello schermo da catturare
                 output_dir="diari"):                # Cartella output
        self.model = model
        self.lmstudio_url = lmstudio_url
        self.capture_interval = capture_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Screen capture
        self.sct = mss.mss()
        if monitor_area:
            self.monitor = monitor_area
        else:
            self.monitor = {"top": 270, "left": 10, "width": 900, "height": 520}

        # Cattura finestra Xiaomi Home (macOS)
        self._xiaomi_window_id = None
        self._use_window_capture = False  # disabilitato — usa screen capture
        if self._use_window_capture:
            self._xiaomi_window_id = self._find_xiaomi_window()
            if self._xiaomi_window_id:
                print(f"[INIT] Cattura finestra Xiaomi Home (window ID: {self._xiaomi_window_id})")
            else:
                print("[INIT] Finestra Xiaomi Home non trovata, uso screen capture")

        # Stato giornata
        self.today = date.today().isoformat()
        self.diary_generated = False

        # Osservazioni grezze (livello 1)
        self.observations = []

        # Riepiloghi orari (livello 2)
        self.hourly_summaries = []

        # Change detection con soglia adattiva
        self._prev_frame_gray = None
        self._prev_observation_time = 0
        self._diff_history = deque(maxlen=20)
        self._change_streak = 0
        self._last_diff = 0

        # Intervalli adattivi
        self._min_interval = capture_interval
        self._current_interval = capture_interval
        self._no_change_streak = 0
        self._prev_logged_interval = capture_interval  # per log cambio intervallo

        # Tracking orario
        self._last_hourly_summary = datetime.now().hour

        # Osservazione di confronto
        self._last_comparison_time = time.time()
        self._comparison_interval = 3600         # ogni 1 ora
        self._comparison_frame = None
        self._comparison_frame_time = None

        # Rilevamento assenza
        self._consecutive_absence = 0
        self._absence_alerted = False
        self._absence_start_time = 0 

        # Prompt di sistema
        self.system_prompt = (
            "Sei un assistente per il monitoraggio domiciliare di una persona anziana.\n\n"
            "Nell'ambiente possono essere presenti una o più persone, "
            "concentrati sulla persona anziana se visibile.\n"
            "Analizza l'immagine e descrivi SOLO informazioni rilevanti dal punto di vista clinico.\n\n"
            "In ogni risposta, valuta sempre:\n"
            "- Presenza o assenza della persona\n"
            "- Postura (seduta, in piedi, sdraiata)\n"
            "- Attività in corso\n"
            "- Stabilità e movimento (normale, lento, incerto)\n"
            "- Se la persona è assistita da qualcuno nell'alzarsi, camminare o altre azioni\n"
            "- Possibili segnali di rischio (caduta, immobilità prolungata, difficoltà)\n\n"
            "Se nelle osservazioni precedenti la persona era in una posizione diversa, "
            "segnala il cambiamento.\n\n"
            "Scrivi in italiano, massimo 2-3 frasi, stile cartella clinica.\n"
            "Sii oggettivo e preciso.\n"
            "NON descrivere dettagli irrilevanti (arredamento, luce, ecc.) "
            "a meno che non siano importanti per la sicurezza.\n"
            "Se la persona non è visibile, dichiaralo chiaramente."
        )

        # Carica dati esistenti se il programma viene riavviato
        self._load_existing_data()

    # =========================================
    # CATTURA FRAME
    # =========================================
    def _find_xiaomi_window(self):
        """Trova l'ID della finestra Xiaomi Home su macOS."""
        if not HAS_QUARTZ:
            return None
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
        for w in windows:
            name = w.get('kCGWindowOwnerName', '')
            if 'Xiaomi' in name:
                return w.get('kCGWindowNumber')
        return None

    def _capture_frame(self):
        """Cattura un frame dalla finestra Xiaomi Home o dallo schermo."""
        if self._use_window_capture:
            frame = self._capture_window()
            if frame is not None:
                return frame
            self._xiaomi_window_id = self._find_xiaomi_window()
            if self._xiaomi_window_id:
                frame = self._capture_window()
                if frame is not None:
                    return frame

        # Fallback: screen capture
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _capture_window(self):
        """Cattura la finestra Xiaomi Home tramite le API macOS."""
        if self._xiaomi_window_id is None:
            return None
        try:
            image = CGWindowListCreateImage(
                CGRectNull,
                kCGWindowListOptionIncludingWindow,
                self._xiaomi_window_id,
                kCGWindowImageDefault
            )
            if image is None:
                self._xiaomi_window_id = None
                return None
            width = Quartz.CGImageGetWidth(image)
            height = Quartz.CGImageGetHeight(image)
            bytes_per_row = Quartz.CGImageGetBytesPerRow(image)
            pixel_data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
            frame = np.frombuffer(pixel_data, dtype=np.uint8)
            frame = frame.reshape(height, bytes_per_row // 4, 4)
            frame = frame[:height, :width, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"[CAPTURE] Errore cattura finestra: {e}")
            self._xiaomi_window_id = None
            return None

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
        
        Soglia adattiva + mini-storia (2 frame consecutivi sopra soglia).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return True

        diff = np.mean(np.abs(gray.astype(float) - self._prev_frame_gray.astype(float)))
        print(f"[DIFF] {diff:.2f}", end=' | ') # log diff per debug
        self._prev_frame_gray = gray
        self._last_diff = diff

        # Soglia adattiva
        self._diff_history.append(diff)
        if len(self._diff_history) >= 5:
            mean_diff = np.mean(list(self._diff_history))
            std_diff = np.std(list(self._diff_history))
            threshold = max(5, mean_diff + 2 * std_diff)
        else:
            threshold = 5

        # Mini-storia: 2 frame consecutivi sopra soglia
        if diff > threshold:
            self._change_streak += 1
        else:
            self._change_streak = 0

        return self._change_streak >= 2

    def _capture_burst(self, n_frames=4, interval=0.5):
        """Cattura una sequenza rapida di frame per analizzare un'azione."""
        frames = []
        for i in range(n_frames):
            frame = self._capture_frame()
            frames.append(self._frame_to_base64(frame))
            if i < n_frames - 1:
                time.sleep(interval)
        return frames

    # =========================================
    # CONTESTO ORARIO
    # =========================================
    def _get_time_context(self):
        """Contesto clinico basato sull'ora del giorno."""
        hour = datetime.now().hour
        if 23 <= hour or hour < 6:
            return (
                "È notte. La persona dovrebbe essere a riposo. "
                "Un'alzata breve può indicare un bisogno fisiologico, "
                "ma attività prolungata o instabilità sono potenzialmente anomale."
            )
        elif 6 <= hour < 8:
            return "È mattina presto, orario tipico del risveglio. Valuta stabilità nei movimenti."
        elif 8 <= hour < 12:
            return "È mattina. Valuta continuità delle attività e mobilità."
        elif 12 <= hour < 14:
            return "È ora di pranzo. Verifica presenza e autonomia."
        elif 14 <= hour < 18:
            return "È pomeriggio. Osserva eventuale sonnolenza o difficoltà nei movimenti."
        elif 18 <= hour < 20:
            return "È ora di cena. Valuta attività e autonomia."
        elif 20 <= hour < 23:
            return "È sera, orario pre-riposo. Osserva stabilità nei movimenti."
        return ""

    # =========================================
    # INTERVALLO ADATTIVO
    # =========================================
    def _update_interval(self, scene_changed):
        """Adatta l'intervallo in base al livello di movimento.
        
        Molto movimento (diff > 15): controlla ogni 10s
        Movimento moderato (diff > 5): controlla ogni 20s
        Scena stabile: intervallo cresce fino a 5min (giorno) o 15min (notte)
        """
        if scene_changed:
            self._no_change_streak = 0
            # Adatta in base all'intensità del cambiamento
            if self._last_diff > 15:
                self._current_interval = 10   # molto movimento
            elif self._last_diff > 5:
                self._current_interval = 20   # movimento moderato
            else:
                self._current_interval = self._min_interval
        else:
            self._no_change_streak += 1
            # Raddoppia ogni 5 frame stabili (non 3 — crescita più lenta)
            if self._no_change_streak % 5 == 0:
                max_int=300
                self._current_interval = min(
                    self._current_interval * 2,
                    max_int
                )

    # =========================================
    # DECISIONE: CHIAMARE IL VLM?
    # =========================================
    def _should_observe(self, scene_changed):
        """Decide se e come osservare.
        
        Ritorna:
            None: non osservare
            'single': frame singolo (scena stabile, check periodico)
            'burst': sequenza standard (4 frame ogni 0.5s)
            'burst_fast': sequenza rapida (5 frame ogni 0.3s, per movimenti veloci)
        """
        now = time.time()
        time_since_last = now - self._prev_observation_time

        if scene_changed and time_since_last >= 15: #time_since_last >= 15 per evitare burst troppo frequenti
            if self._last_diff > 15:
                return 'burst_fast'
            return 'burst'
        if time_since_last >= self._current_interval: 
            return 'single'

        return None #se non c'è movimento o non è passato abbastanza tempo dall'ultima osservazione

    # =========================================
    # CHIAMATA VLM
    # =========================================
    def _call_vlm(self, images_b64, context_messages=None, max_tokens=200, prompt_text=None):
        """Invia una o più immagini a LM Studio con contesto."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context_messages:
            messages.extend(context_messages)

        now = datetime.now().strftime("%H:%M:%S")
        time_ctx = self._get_time_context()

        if isinstance(images_b64, list):
            if prompt_text is None:
                prompt_text = (
                    f"Ore {now}. {time_ctx} "
                    f"Questa è una sequenza di {len(images_b64)} frame consecutivi "
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
            if prompt_text is None:
                prompt_text = f"Ore {now}. {time_ctx} Descrivi cosa vedi."
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
    """
    def _is_redundant(self, description):
        #Evita osservazioni ripetitive (stesso stato 3+ volte consecutive).
        if len(self.observations) < 2:
            return False
        
        recent = [o['description'].lower() for o in self.observations[-2:]]
        current = description.lower()
        
        states = ['seduta', 'in piedi', 'cammina', 'sdraiata', 
                'non visibile', 'non è visibile', 'non presente']
        for state in states:
            if state in current and all(state in d for d in recent):
                return True
        return False
    """
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
    # CONTESTO CONVERSAZIONALE
    # =========================================
    def _build_context(self):
        """Contesto intelligente: ultime osservazioni rilevanti."""
        if not self.observations:
            return None

        recent = self.observations[-3:]
        summary = "Osservazioni precedenti:\n"
        for obs in recent:
            obs_type = obs.get('type', 'singolo') #se non è presente il campo 'type' nell'osservazione, assume che sia 'singolo' per mantenere la compatibilità con eventuali osservazioni salvate in precedenza senza questo campo.
            tag = ""
            if obs_type == "alert":
                tag = " [ALERT]"
            elif obs_type == "confronto":
                tag = " [CONFRONTO]"
            summary += f"- Ore {obs['time']}{tag}: {obs['description']}\n"

        return [{"role": "user", "content": summary + "\nOra osserva il frame corrente."}] 
        """
        #il contesto è costruito come un messaggio utente che riassume le ultime 3 
        # osservazioni, con eventuali tag per alert o confronti, seguito da 
        # un invito a osservare il frame corrente. 
        # Questo aiuta il VLM ad avere una memoria a breve termine delle osservazioni recenti
        #  e a contestualizzare meglio la sua analisi del nuovo frame.
        """

    # =========================================
    # LIVELLO 1: OSSERVAZIONE
    # =========================================
    def _observe(self, frame, mode='single'):
        """Analizza il frame (o una sequenza) e salva l'osservazione."""
        context = self._build_context()

        if mode == 'burst_fast':
            images = self._capture_burst(n_frames=5, interval=0.3)
            description = self._call_vlm(images, context, max_tokens=250)
            obs_type = "sequenza_rapida"
            n_frames = 5
        elif mode == 'burst':
            images = self._capture_burst(n_frames=4, interval=0.5)
            description = self._call_vlm(images, context, max_tokens=250)
            obs_type = "sequenza"
            n_frames = 4
        else:
            image_b64 = self._frame_to_base64(frame)
            description = self._call_vlm(image_b64, context)
            obs_type = "singolo"
            n_frames = 1

        if description:
            """
            if self._is_redundant(description):
                self._prev_observation_time = time.time()
                print(f"[{datetime.now().strftime('%H:%M')}] [SKIP] Stato invariato")
                return False
            """
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

            # Tag con numero di frame per il log
            if mode == 'burst_fast':
                tag = f"FAS×{n_frames}"
            elif mode == 'burst':
                tag = f"SEQ×{n_frames}"
            else:
                tag = f"   ×{n_frames}"
            print(f"[{obs['time']}] [{tag}] {description}")

            self._track_absence(description)
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] Nessuna risposta dal VLM")
            return False

    # =========================================
    # RILEVAMENTO ASSENZA PROLUNGATA
    # =========================================
    def _track_absence(self, description):
        """Traccia osservazioni consecutive senza persona visibile."""
        desc_lower = description.lower() #il metodo lower() converte la stringa description in minuscolo, in modo da rendere il controllo case-insensitive. In questo modo, se il VLM risponde con "Non è visibile" o "non è visibile", entrambe le risposte saranno riconosciute come indicazioni di assenza della persona.
        person_absent = ("non è visibile" in desc_lower or
                         "non visibile" in desc_lower or
                         "assenza" in desc_lower or
                         "non è presente" in desc_lower or
                         "non presente" in desc_lower)

        if person_absent:
            if self._consecutive_absence == 0:
                self._absence_start_time = time.time()  # inizio assenza
            self._consecutive_absence += 1
        else:
            self._consecutive_absence = 0
            self._absence_alerted = False
        if self._consecutive_absence > 0:
            minutes_absent = (time.time() - self._absence_start_time) / 60
        else:
            minutes_absent = 0
        if (minutes_absent >= 30 and not self._absence_alerted):
            alert_obs = {
                "time": datetime.now().strftime("%H:%M"),
                "timestamp": datetime.now().isoformat(),
                "hour": datetime.now().hour,
                "type": "alert",
                "description": (f"⚠ ALERT: La persona non è visibile da circa "
                               f"{minutes_absent:.0f} minuti durante le ore diurne. "
                               f"Verificare se è uscita o se si trova fuori dall'inquadratura.")
            }
            self.observations.append(alert_obs)
            self._save_data()
            self._absence_alerted = True
            print(f"\n{'!'*60}")
            print(f"[{alert_obs['time']}] {alert_obs['description']}")
            print(f"{'!'*60}\n")

    # =========================================
    # OSSERVAZIONE DI CONFRONTO
    # =========================================
    def _check_comparison(self, frame):
        """Ogni ora confronta il frame corrente con quello precedente."""
        now = time.time()
        if now - self._last_comparison_time < self._comparison_interval:
            return

        self._last_comparison_time = now

        if self._comparison_frame is None:
            self._comparison_frame = self._frame_to_base64(frame) #frame precedente per il confronto
            self._comparison_frame_time = datetime.now().strftime("%H:%M") #salva l'ora del frame di confronto
            return

        current_b64 = self._frame_to_base64(frame) #frame corrente
        now_str = datetime.now().strftime("%H:%M") #ottiene l'ora corrente in formato stringa per usarla nel prompt e nell'osservazione

        prompt = (
            f"Ti mostro due immagini della stessa stanza. "
            f"La prima è delle ore {self._comparison_frame_time}, la seconda delle ore {now_str}. "
            f"Ignora la persona. Concentrati sull'AMBIENTE: "
            f"ci sono oggetti caduti, ostacoli nuovi, sedie spostate, "
            f"o qualsiasi cambiamento che potrebbe rappresentare un rischio? "
            f"Se non noti cambiamenti rilevanti, scrivi 'Ambiente invariato'."
            f"Nota che l'inquadratura potrebbe essere cambiata."
        )

        images = [self._comparison_frame, current_b64]
        context = self._build_context()
        description = self._call_vlm(images, context, max_tokens=250, prompt_text=prompt)

        if description:
            obs = {
                "time": now_str,
                "timestamp": datetime.now().isoformat(),
                "hour": datetime.now().hour,
                "type": "confronto",
                "description": f"[CONFRONTO {self._comparison_frame_time}→{now_str}] {description}"
            }
            self.observations.append(obs)
            self._save_data()
            print(f"[{now_str}] [CMP] {description}")

        self._comparison_frame = current_b64
        self._comparison_frame_time = now_str

    # =========================================
    # LIVELLO 2: SINTESI ORARIA
    # =========================================
    def _generate_hourly_summary(self, hour):
        """Genera un riepilogo per un'ora specifica."""
        hour_obs = [o for o in self.observations if o.get('hour') == hour] #prende tutte le osservazioni dell'ora specifica
        if not hour_obs:
            return

        existing_hours = {s['hour'] for s in self.hourly_summaries} # per evitare riepiloghi duplicati se il programma viene riavviato
        if hour in existing_hours:
            return

        obs_text = "\n".join(f"- {o['time']}: {o['description']}" for o in hour_obs)
        """
        ogni osservazione dell'ora viene formattata come "- ore: descrizione" e concatenata in un 
        unico testo che sarà usato come input per il VLM per generare la sintesi oraria.
        """

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

        path = self.output_dir / f"diario_{self.today}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + diary_text)
        print(f"[DIARIO] Salvato in {path}")

    # =========================================
    # LIVELLO 4: DIARIO SETTIMANALE
    # =========================================
    def generate_weekly_diary(self, end_date=None):
        """Genera il diario settimanale dagli ultimi 7 diari giornalieri."""
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

        diary = self._call_vlm_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare a lungo termine.",
            max_tokens=4000
        )

        if diary:
            self._save_weekly_diary(diary, start_date, end_date, len(daily_diaries))
            return diary
        else:
            print("[SETTIMANALE] Errore nella generazione")
            return None

    def _save_weekly_diary(self, diary_text, start_date, end_date, n_days):
        header = (
            f"REPORT SETTIMANALE DI MONITORAGGIO DOMICILIARE\n"
            f"{'='*50}\n"
            f"Periodo: {start_date.isoformat()} → {end_date.isoformat()}\n"
            f"Giorni con dati: {n_days}/7\n"
            f"{'='*50}\n\n"
        )
        path = self.output_dir / f"settimanale_{start_date.isoformat()}_{end_date.isoformat()}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + diary_text)
        print(f"[SETTIMANALE] Salvato in {path}")

    # =========================================
    # LIVELLO 5: DIARIO MENSILE
    # =========================================
    def generate_monthly_diary(self, year=None, month=None):
        """Genera il diario mensile dai report settimanali o diari giornalieri."""
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
            content = "\n\n".join(
                f"--- Settimana {entry['period']} ---\n{entry['content']}"
                for entry in weekly_reports
            )
            source = f"{len(weekly_reports)} report settimanali"
        else:
            content = "\n\n".join(
                f"--- {entry['date']} ---\n{entry['content']}"
                for entry in daily_diaries
            )
            source = f"{len(daily_diaries)} diari giornalieri"

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

        diary = self._call_vlm_text(
            prompt,
            system="Sei un geriatra esperto in monitoraggio domiciliare a lungo termine.",
            max_tokens=5000
        )

        if diary:
            self._save_monthly_diary(diary, year, month, start_date, end_date)
            return diary
        else:
            print("[MENSILE] Errore nella generazione")
            return None

    def _save_monthly_diary(self, diary_text, year, month, start_date, end_date):
        header = (
            f"REPORT MENSILE DI MONITORAGGIO DOMICILIARE\n"
            f"{'='*50}\n"
            f"Mese: {start_date.strftime('%B %Y')}\n"
            f"Periodo: {start_date.isoformat()} → {end_date.isoformat()}\n"
            f"{'='*50}\n\n"
        )
        path = self.output_dir / f"mensile_{year}-{month:02d}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + diary_text)
        print(f"[MENSILE] Salvato in {path}")

    # =========================================
    # LETTURA DIARI PRECEDENTI
    # =========================================
    def _read_daily_diaries(self, start_date, end_date):
        """Legge i diari giornalieri nel range di date."""
        diaries = []
        current = start_date
        while current <= end_date:
            path = self.output_dir / f"diario_{current.isoformat()}.txt"
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
        for path in sorted(self.output_dir.glob("settimanale_*.txt")):
            try:
                parts = path.stem.replace("settimanale_", "").split("_")
                week_start = date.fromisoformat(parts[0])
                week_end = date.fromisoformat(parts[1])
                if week_start <= end_date and week_end >= start_date:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    body_parts = content.split('\n\n', 1)
                    body = body_parts[1] if len(body_parts) > 1 else content
                    reports.append({
                        "period": f"{week_start.isoformat()} → {week_end.isoformat()}",
                        "content": body.strip()
                    })
            except (ValueError, IndexError):
                continue
        print(f"[LETTURA] Trovati {len(reports)} report settimanali nel periodo")
        return reports

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
        today = date.today()
        today_str = today.isoformat()
        if today_str != self.today: #serve per capire quando è cambiato il giorno e quindi generare il diario
            self._generate_hourly_summary(self._last_hourly_summary)

            if not self.diary_generated and self.observations: 
                print(f"[DIARIO] Cambio giornata, genero diario per {self.today}")
                self.generate_diary()

            if today.weekday() == 0:
                yesterday = today - timedelta(days=1)
                print(f"[SETTIMANALE] È lunedì, genero report settimanale")
                self.generate_weekly_diary(end_date=yesterday)

            if today.day == 1:
                print(f"[MENSILE] Primo del mese, genero report mensile")
                self.generate_monthly_diary()

            self.today = today_str
            self.observations = []
            self.hourly_summaries = []
            self.diary_generated = False
            self._last_hourly_summary = datetime.now().hour
            self._consecutive_absence = 0
            self._absence_alerted = False
            self._comparison_frame = None
            self._load_existing_data()
            print(f"[INIT] Nuovo giorno: {self.today}")

    # =========================================
    # ANTEPRIMA
    # =========================================
    def preview(self):
        """Mostra una finestra con l'area catturata. Premi Q per chiudere."""
        if self._xiaomi_window_id:
            print(f"[PREVIEW] Cattura finestra Xiaomi Home (window ID: {self._xiaomi_window_id})")
        else:
            print(f"[PREVIEW] Screen capture - Area: top={self.monitor['top']} left={self.monitor['left']} "
                  f"{self.monitor['width']}x{self.monitor['height']}")
        print("Premi Q per chiudere l'anteprima e avviare il monitoraggio\n")

        cv2.namedWindow("VLM Monitor - Preview (Q per chiudere)", cv2.WINDOW_NORMAL)
        cv2.moveWindow("VLM Monitor - Preview (Q per chiudere)", 1000, 0)

        while True:
            frame = self._capture_frame()
            cv2.imshow("VLM Monitor - Preview (Q per chiudere)", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    # =========================================
    # LOOP PRINCIPALE
    # =========================================
    def run(self):
        print(f"{'='*60}")
        capture_mode = "Finestra Xiaomi Home" if self._xiaomi_window_id else "Screen capture"
        print(f"VLM Daily Monitor — Monitoraggio h24")
        print(f"  Modello:      {self.model}")
        print(f"  Server:       {self.lmstudio_url}")
        print(f"  Cattura:      {capture_mode}")
        print(f"  Intervallo:   {self.capture_interval}s (adattivo)")
        print(f"  Diario:       a mezzanotte (automatico)")
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

                # Log cambio intervallo
                if self._current_interval != self._prev_logged_interval: #senza questo controllo, ad ogni ciclo in cui cambia l'intervallo, viene stampato il messaggio di cambio intervallo, anche se l'intervallo è già stato cambiato in precedenza. Con questo controllo, il messaggio viene stampato solo quando l'intervallo effettivamente cambia rispetto all'ultimo intervallo registrato.
                    print(f"[INTERVAL] {self._prev_logged_interval}s → {self._current_interval}s "
                          f"({'movimento' if changed else 'stabile'})")
                    self._prev_logged_interval = self._current_interval

                obs_mode = self._should_observe(changed)
                if obs_mode:
                    self._observe(frame, mode=obs_mode)

                self._check_comparison(frame)

                time.sleep(10) 

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
    parser.add_argument("--output", default="diari")
    parser.add_argument("--top", type=int, default=270)
    parser.add_argument("--left", type=int, default=10)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--preview", action="store_true",
                        help="Mostra anteprima dell'area catturata prima di avviare")
    parser.add_argument("--gen-weekly", action="store_true",
                        help="Genera il report settimanale e esci")
    parser.add_argument("--gen-monthly", action="store_true",
                        help="Genera il report mensile e esci")

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
        monitor.generate_weekly_diary()
        return
    if args.gen_monthly:
        monitor.generate_monthly_diary()
        return
    if args.preview:
        monitor.preview()

    monitor.run()


if __name__ == "__main__":
    main()