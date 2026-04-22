"""
VLM Daily Monitor — Monitoraggio h24 con Vision Language Model.

Architettura a tre livelli:
  1. Cattura intelligente: confronta i frame e chiama il VLM solo quando
     la scena cambia o è passato troppo tempo dall'ultima osservazione.
  2. Sintesi oraria: ogni ora condensa le osservazioni in un paragrafo.
  3. Diario giornaliero: a fine giornata, sintetizza i riepiloghi orari
     in un diario narrativo di 2-3 pagine.

Migliorie:
  - Soglia diff adattiva (media + 2σ dei diff recenti)
  - Mini-storia: servono 2+ diff consecutivi sopra soglia per confermare un cambiamento
  - Smart burst: burst veloce per movimenti rapidi, lento per movimenti graduali
  - Prompt con contesto orario (notte vs giorno)
  - Osservazione di confronto ogni 2 ore
  - Rilevamento assenza prolungata

Uso:
    python vlm_monitor.py
    python vlm_monitor.py --preview
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
from collections import deque


class VLMMonitor:
    def __init__(self,
                 model="gemma-4-26b-a4b-it",       # Modello VLM da LM Studio
                 lmstudio_url="http://localhost:1234", # URL del server locale LM Studio
                 capture_interval=30,                # Intervallo base in secondi (adattivo)
                 monitor_area=None,                  # Area dello schermo da catturare
                 diary_hour=22,                      # Ora generazione diario
                 output_dir="diari"):                # Cartella output
        self.model = model
        self.lmstudio_url = lmstudio_url
        self.capture_interval = capture_interval
        self.diary_hour = diary_hour
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Screen capture
        self.sct = mss.mss()
        if monitor_area:
            self.monitor = monitor_area
        else:
            self.monitor = {"top": 270, "left": 10, "width": 900, "height": 520}

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
        self._diff_history = deque(maxlen=20)   # storico dei diff per soglia adattiva
        self._change_streak = 0                  # frame consecutivi con cambiamento
        self._last_diff = 0                      # grandezza ultimo diff (per smart burst)

        # Intervalli adattivi
        self._min_interval = capture_interval
        self._max_interval = 900                 # 15 min se scena stabile (notte)
        self._current_interval = capture_interval
        self._no_change_streak = 0

        # Tracking orario
        self._last_hourly_summary = datetime.now().hour

        # Osservazione di confronto
        self._last_comparison_time = time.time()
        self._comparison_interval = 7200         # ogni 2 ore
        self._comparison_frame = None            # frame di riferimento per il confronto
        self._comparison_frame_time = None

        # Rilevamento assenza
        self._consecutive_absence = 0            # osservazioni consecutive senza persona
        self._absence_alerted = False            # evita alert ripetuti

        # Prompt di sistema
        self.system_prompt = (
            "Sei un assistente per il monitoraggio domiciliare di una persona anziana.\n\n"
            "Nell'ambiente possono essere presenti una o più persone, concentrati sulla persona anziana se visibile"
            "Analizza l'immagine e descrivi SOLO informazioni rilevanti dal punto di vista clinico.\n\n"
            "In ogni risposta, valuta sempre:\n"
            "- Presenza o assenza della persona\n"
            "- Postura (seduta, in piedi, sdraiata)\n"
            "- Attività in corso\n"
            "- Stabilità e movimento (normale, lento, incerto)\n"
            "-se la persona è assista da qualcuno nell'alzarsi, camminare o altre azioni quotidiane\n"
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
        
        Usa soglia adattiva: media dei diff recenti + 2 deviazioni standard.
        Di notte (scena stabile, diff medio 1-2) la soglia scende a ~5.
        Di giorno (luce variabile, diff medio 3-4) sale a ~8-10.
        
        Richiede 2 frame consecutivi sopra soglia per confermare il cambiamento,
        filtrando ombre, riflessi e rumore momentaneo.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return True

        diff = np.mean(np.abs(gray.astype(float) - self._prev_frame_gray.astype(float)))
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
        """Cattura una sequenza rapida di frame per analizzare un'azione.
        
        Args:
            n_frames: quanti frame catturare
            interval: secondi tra i frame
        """
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
        """Ritorna una frase di contesto basata sull'ora del giorno.
        risulta utile per guidare il VLM a interpretare meglio la scena
        
        Di notte qualsiasi attività è rilevante.
        A pranzo il contesto è diverso dalla mattina.
        """
        hour = datetime.now().hour
        if 23 <= hour or hour < 6:
            return "È notte. Qualsiasi attività in questo orario è potenzialmente rilevante. La persona dovrebbe essere a riposo."
        elif 6 <= hour < 8:
            return "È mattina presto, orario tipico del risveglio. valuta se la persona mantiene stabilità"
        elif 8 <= hour < 12:
            return "È mattina. valuta anche il tipo di attività"
        elif 12 <= hour < 14:
            return "È ora di pranzo."
        elif 14 <= hour < 18:
            return "È pomeriggio. Osserva eventuale difficoltà nei movimenti"
        elif 18 <= hour < 20:
            return "È ora di cena."
        elif 20 <= hour < 23:
            return "È sera, orario pre-riposo."
        return ""

    # =========================================
    # INTERVALLO ADATTIVO
    # =========================================
    def _update_interval(self, scene_changed):
        """Adatta l'intervallo: attività → 30s, stabile/notte → fino a 15 min."""
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
        
        Ritorna:
            None: non osservare
            'single': frame singolo (scena stabile, check periodico)
            'burst': sequenza standard (4 frame ogni 0.5s)
            'burst_fast': sequenza rapida (5 frame ogni 0.3s, per movimenti veloci)
        """
        now = time.time()
        time_since_last = now - self._prev_observation_time

        # Scena cambiata → burst
        if scene_changed and time_since_last >= 15:
            # Diff grande (>15): movimento rapido → burst veloce
            if self._last_diff > 15:
                return 'burst_fast'
            # Diff medio: movimento normale → burst standard
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
            prompt_text: testo personalizzato
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if context_messages:
            messages.extend(context_messages)

        now = datetime.now().strftime("%H:%M:%S")
        time_ctx = self._get_time_context()

        # Costruisci il contenuto con una o più immagini
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

        # Prendi le ultime 3 osservazioni, ma solo il testo
        # Non costruire una finta conversazione — dai un riepilogo
        recent = self.observations[-3:]
        
        summary = "Osservazioni precedenti:\n"
        for obs in recent:
            obs_type = obs.get('type', 'singolo')
            tag = ""
            if obs_type == "alert":
                tag = " [ALERT]"
            elif obs_type == "confronto":
                tag = " [CONFRONTO]"
            summary += f"- Ore {obs['time']}{tag}: {obs['description']}\n"

        return [{"role": "user", "content": summary + "\nOra osserva il frame corrente."}]
    

    # =========================================
    # LIVELLO 1: OSSERVAZIONE
    # =========================================
    def _observe(self, frame, mode='single'):
        """Analizza il frame (o una sequenza) e salva l'osservazione.
        
        Args:
            frame: frame corrente (usato per 'single')
            mode: 'single', 'burst', o 'burst_fast'
        """
        context = self._build_context()

        if mode == 'burst_fast':
            # Movimento rapido → 5 frame ravvicinati
            images = self._capture_burst(n_frames=5, interval=0.3)
            description = self._call_vlm(images, context, max_tokens=250)
            obs_type = "sequenza_rapida"
        elif mode == 'burst':
            # Movimento normale → 4 frame standard
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

            # Tag per il log
            tags = {"singolo": "   ", "sequenza": "SEQ", "sequenza_rapida": "FAS"}
            tag = tags.get(obs_type, "   ")
            print(f"[{obs['time']}] [{tag}] {description}")

            # Traccia assenza
            self._track_absence(description)

            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] Nessuna risposta dal VLM")
            return False

    # =========================================
    # RILEVAMENTO ASSENZA PROLUNGATA
    # =========================================
    def _track_absence(self, description):
        """Traccia osservazioni consecutive senza persona visibile.
        
        Se la persona non è visibile per 30+ minuti durante il giorno,
        genera un'osservazione di alert.
        """
        desc_lower = description.lower()
        person_absent = ("non è visibile" in desc_lower or
                         "non visibile" in desc_lower or
                         "assenza" in desc_lower or
                         "non è presente" in desc_lower or
                         "non presente" in desc_lower)

        if person_absent:
            self._consecutive_absence += 1
        else:
            self._consecutive_absence = 0
            self._absence_alerted = False

        # Alert dopo ~30 minuti di assenza durante il giorno (6:00-22:00)
        # Con intervallo adattivo, 30 min ~ 2-6 osservazioni consecutive
        hour = datetime.now().hour
        is_daytime = 6 <= hour < 22
        minutes_absent = self._consecutive_absence * self._current_interval / 60

        if (is_daytime and minutes_absent >= 30 and not self._absence_alerted):
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
        """Ogni 2 ore manda al VLM il frame corrente e quello di 2 ore prima.
        Cattura cambiamenti graduali che il diff frame-to-frame non vede:
        persona che si accascia lentamente, oggetto caduto, cambio di postura.
        """
        now = time.time()
        if now - self._last_comparison_time < self._comparison_interval:
            return

        self._last_comparison_time = now

        if self._comparison_frame is None:
            # Primo frame di riferimento
            self._comparison_frame = self._frame_to_base64(frame)
            self._comparison_frame_time = datetime.now().strftime("%H:%M")
            return

        # Manda entrambi i frame
        current_b64 = self._frame_to_base64(frame)
        now_str = datetime.now().strftime("%H:%M")

        prompt = (
            f"Ti mostro due immagini della stessa stanza. "
            f"La prima è delle ore {self._comparison_frame_time}, la seconda delle ore {now_str}. "
            f"Ci sono cambiamenti nello stato della persona o nell'ambiente? "
            f"La persona è nella stessa posizione? Si è spostata? "
            f"Noti oggetti caduti, cambiamenti nella postura, o qualsiasi cosa diversa? "
            f"Rispondi in 2-3 frasi."
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

        # Aggiorna il frame di riferimento
        self._comparison_frame = current_b64
        self._comparison_frame_time = now_str

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

        # Conta gli alert
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

        # Conta tipi di osservazione
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
            self._consecutive_absence = 0
            self._absence_alerted = False
            self._comparison_frame = None
            self._load_existing_data()
            print(f"[INIT] Nuovo giorno: {self.today}")

    # =========================================
    # ANTEPRIMA AREA DI CATTURA
    # =========================================
    def preview(self):
        """Mostra una finestra con l'area catturata. Premi Q per chiudere."""
        print(f"[PREVIEW] Area: top={self.monitor['top']} left={self.monitor['left']} "
              f"{self.monitor['width']}x{self.monitor['height']}")
        print("Premi Q per chiudere l'anteprima e avviare il monitoraggio\n")

        cv2.namedWindow("VLM Monitor - Preview (Q per chiudere)", cv2.WINDOW_NORMAL)
        cv2.moveWindow("VLM Monitor - Preview (Q per chiudere)",
                       self.monitor['left'] + self.monitor['width'] + 10, 0)

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

                # Osservazione regolare o burst
                obs_mode = self._should_observe(changed)
                if obs_mode:
                    self._observe(frame, mode=obs_mode)

                # Osservazione di confronto ogni 2 ore
                self._check_comparison(frame)

                # Genera diario all'ora impostata
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
    parser.add_argument("--top", type=int, default=270)
    parser.add_argument("--left", type=int, default=10)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--preview", action="store_true",
                        help="Mostra anteprima dell'area catturata prima di avviare")

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

    if args.preview:
        monitor.preview()

    monitor.run()


if __name__ == "__main__":
    main()