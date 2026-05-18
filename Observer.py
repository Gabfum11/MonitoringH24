"""
Modulo di osservazione.

Gestisce:
- Decisione se/come osservare (single, sequenza, sequenza_rapida)
- Intervallo adattivo basato sul livello di movimento
- Filtro ridondanza per evitare osservazioni ripetitive
- Contesto conversazionale per il VLM
- Rilevamento assenza prolungata con alert
- Osservazioni di confronto ambientale periodiche
"""

import time
from datetime import datetime


class Observer:
    def __init__(self, capture_manager, vlm_client, observations,
                 save_callback, capture_interval=30, comparison_interval=1800):
        """
        Args:
            capture_manager: istanza di CaptureManager
            vlm_client: istanza di VLMClient
            observations: lista condivisa delle osservazioni (passata per riferimento)
            save_callback: funzione da chiamare per salvare i dati su disco
            capture_interval: intervallo base in secondi
            comparison_interval: secondi tra confronti ambientali (default: 30 minuti)
        """
        self.capture = capture_manager
        self.vlm = vlm_client
        self.observations = observations
        self._save = save_callback

        # Intervalli adattivi
        self._min_interval = capture_interval
        self._current_interval = capture_interval
        self._no_change_streak = 0
        self._prev_observation_time = 0
        self._prev_logged_interval = capture_interval

        # Assenza
        self._consecutive_absence = 0
        self._absence_alerted = False
        self._absence_start_time = 0
        self._zoom_pending = False

        # Confronto ambientale
        self._last_comparison_time = time.time()
        self._comparison_interval = comparison_interval
        self._comparison_frame = None
        self._comparison_frame_time = None

        # Contesto orario (impostato da Monitor ad ogni nuova sintesi oraria)
        self._last_hourly_text = None

    @property 
    def current_interval(self):
        return self._current_interval

    def reset(self):
        """Reset giornaliero dello stato."""
        self._consecutive_absence = 0
        self._absence_alerted = False
        self._comparison_frame = None

    # =========================================
    # INTERVALLO ADATTIVO
    # =========================================
    def update_interval(self, scene_changed, last_diff):
        """
        Adatta l'intervallo di osservazione VLM.
        Il loop principale gira ogni 2s, qui decidiamo ogni quanto analizzare.
        """
        if scene_changed:
            self._no_change_streak = 0
            # Se c'è movimento importante, vogliamo essere pronti a scattare spesso
            if last_diff > 15:
                self._current_interval = 20  # Check ogni 10s se il movimento persiste
            elif last_diff > 5:
                self._current_interval = 30
            else:
                self._current_interval = self._min_interval # Default (es. 30s)
        else:
            # La scena è ferma. Aumentiamo l'attesa per risparmiare risorse.
            self._no_change_streak += 1
            
            # Ogni 5 cicli di stabilità (ovvero ogni 10 secondi reali)
            # aumentiamo l'intervallo di controllo.
            if self._no_change_streak % 5 == 0:
                self._current_interval = min(
                    self._current_interval + 20, # Saliamo gradualmente
                    120  # Max 2 minuti tra un check e l'altro se non succede nulla
                )

        # Log critico per capire se il "cervello" sta accelerando o rallentando
        #if self._current_interval != self._prev_logged_interval:
            #print(f"[IA-STRATEGY] Prossimo check tra {self._current_interval}s "
                #f"({'REATTIVO' if scene_changed else 'RISPARMIO'})")
            self._prev_logged_interval = self._current_interval

    # =========================================
    # DECISIONE: CHIAMARE IL VLM?
    # =========================================
    def should_observe(self, scene_changed, last_diff, change_streak=0):
        """Decide se e come osservare.

        sequenza_rapida: azione intensa e sostenuta (diff alto + streak alto)
        sequenza:      movimento normale
        single:     check periodico su scena stabile
        """
        now = time.time()
        time_since_last = now - self._prev_observation_time

        if scene_changed and time_since_last >= 15:
            if last_diff > 6 and change_streak >= 3:
                return 'sequenza_rapida'
            return 'sequenza'

        if time_since_last >= self._current_interval:
            return 'single'

        return None

    # =========================================
    # CONTESTO CONVERSAZIONALE
    # =========================================
    def _build_context(self):
        """Contesto intelligente: ultime  5 osservazioni come riepilogo e il riassunto dell'ultima ora.
        """
        if not self.observations:
            return None

        summary = ""
        if self._last_hourly_text:
            summary += f"Riepilogo dell'ultima ora: {self._last_hourly_text}\n\n"

        # Ultime 5 osservazioni (non 3 — abbiamo spazio)
        recent = self.observations[-5:]
        summary += "Osservazioni recenti:\n"
        for obs in recent:
            obs_type = obs.get('type', 'singolo')
            tag = ""
            if obs_type == "alert": #per evitare di confondere gli alert con le osservazioni normali, aggiungiamo un tag [ALERT] alle osservazioni di tipo alert, così il VLM può dare loro la giusta attenzione e priorità nella generazione della risposta.
                tag = " [ALERT]"
            elif obs_type == "confronto": #stessa cosa per le osservazioni di confronto ambientale, che sono particolarmente importanti per rilevare cambiamenti rischiosi nell'ambiente. Aggiungiamo un tag [CONFRONTO] per evidenziarle nel contesto e far capire al VLM che sono osservazioni chiave da considerare nella sua analisi.
                tag = " [CONFRONTO]"
            summary += f"- Ore {obs['time']}{tag}: {obs['description']}\n"

        return [{"role": "user", "content": summary + "\nOra osserva il frame corrente."}]

    # =========================================
    # OSSERVAZIONE
    # =========================================
    def observe(self, frame, mode='single'):
        """Analizza il frame (o una sequenza), salva l'osservazione e traccia l'assenza.
        
        Args:
            frame: frame corrente (usato per 'single')
            mode: 'single', 'sequenza', o 'sequenza_rapida'
            
        Returns:
            bool: True se l'osservazione è stata salvata, False se skippata/errore
        """
        
        context = self._build_context()
        if isinstance(frame, list):
            images = frame
            obs_type = mode if mode in ("sequenza", "sequenza_rapida") else "sequenza"
            n_frames = len(images)
            description = self.vlm.call_with_images(images, context)
        else: #per le osservazioni singole, se il diff è molto basso e abbiamo già un'osservazione recente, skippiamo per evitare ripetizioni inutili su scena stabile
            if (self.capture.last_diff < 2.0 and
                len(self.observations) > 0 and
                time.time() - self._prev_observation_time < 30): #se c'è stata un osservazione negli ultimi 30 secondi e il diff è molto basso, consideriamo la scena stabile e skippiamo
                self._prev_observation_time = time.time()
                print(f"[{datetime.now().strftime('%H:%M')}] [SKIP] Scena stabile (diff={self.capture.last_diff:.1f})")
                return False
            image_b64 = self.capture.frame_to_base64(frame)
            description = self.vlm.call_with_images(image_b64, context)
            obs_type = "singolo"
            n_frames = 1

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
            self._save()
            tag = "EVT_R" if obs_type == "sequenza_rapida" else ("EVT" if obs_type == "sequenza" else "FIX")
            print(f"[{obs['time']}] [{tag}×{n_frames}] {description}")
            self._track_absence(description)
            self._track_anomalies(description)
            if self._zoom_pending:
                self._zoom_pending = False
                self._observe_zoomed()
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] Nessuna risposta dal VLM")
            return False

    def _observe_zoomed(self):
        """Osservazione con zoom automatico: doppio click → cattura → doppio click."""
        self.capture.zoom_in()
        frame = self.capture.capture_frame()
        image_b64 = self.capture.frame_to_base64(frame)
        context = self._build_context()
        description = self.vlm.call_with_images(image_b64, context)
        self.capture.zoom_out()

        if description:
            obs = {
                "time": datetime.now().strftime("%H:%M"),
                "timestamp": datetime.now().isoformat(),
                "hour": datetime.now().hour,
                "type": "singolo",
                "description": f"[ZOOM] {description}"
            }
            self.observations.append(obs)
            self._save()
            print(f"[{obs['time']}] [ZOOM] {description}")
            self._track_absence(description)

    # =========================================
    # RILEVAMENTO ASSENZA PROLUNGATA
    # =========================================
    def _track_absence(self, description):
        """Traccia osservazioni consecutive senza persona visibile.
        
        Se la persona non è visibile per 30+ minuti durante il giorno (6-22),
        genera un alert e lo aggiunge alle osservazioni.
        """
        desc_lower = description.lower()
        person_absent = ("non è visibile" in desc_lower or
                         "non visibile" in desc_lower or
                         "assenza" in desc_lower or
                         "non è presente" in desc_lower or
                         "non presente" in desc_lower)

        if person_absent:
            if self._consecutive_absence == 0:
                self._absence_start_time = time.time()
            self._consecutive_absence += 1
            self._zoom_pending = True
        else:
            self._consecutive_absence = 0
            self._absence_alerted = False
            self._zoom_pending = False

        # Calcolo basato sul tempo reale
        if self._consecutive_absence > 0:
            minutes_absent = (time.time() - self._absence_start_time) / 60
        else:
            minutes_absent = 0

        hour = datetime.now().hour
        is_daytime = 6 <= hour < 22

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
            self._save()
            self._absence_alerted = True
            print(f"\n{'!'*60}")
            print(f"[{alert_obs['time']}] {alert_obs['description']}")
            print(f"{'!'*60}\n")

    # =========================================
    # RILEVAMENTO ANOMALIE
    # =========================================
    _ALERT_PATTERNS = {
        "caduta": [
            "a terra", "caduta", "è caduta", "è caduto", "distesa sul pavimento",
            "disteso sul pavimento", "giace", "giace sul pavimento", "è scivolata", "è scivolato"
        ],
        "instabilità grave": [
            "barcolla", "perde l'equilibrio", "si aggrappa", "si regge al muro",
            "rischio di caduta", "quasi caduta", "ha perso l'equilibrio"
        ],
        "difficoltà evidente": [
            "in evidente difficoltà", "si contorce", "si tiene la testa",
            "non riesce ad alzarsi", "non riesce a muoversi", "sembra sofferente"
        ],
        "estraneo": [
            "persona sconosciuta", "estraneo", "individuo non identificato",
            "presenza non abituale"
        ],
    }

    def _track_anomalies(self, description):
        """Rileva situazioni critiche nel testo dell'osservazione e genera alert immediati."""
        desc_lower = description.lower()
        for categoria, termini in self._ALERT_PATTERNS.items():
            if any(t in desc_lower for t in termini):
                alert_obs = {
                    "time": datetime.now().strftime("%H:%M"),
                    "timestamp": datetime.now().isoformat(),
                    "hour": datetime.now().hour,
                    "type": "alert",
                    "description": (f"⚠ ALERT ({categoria}): {description}")
                }
                self.observations.append(alert_obs)
                self._save()
                print(f"\n{'!'*60}")
                print(f"[{alert_obs['time']}] ALERT {categoria.upper()}: {description}")
                print(f"{'!'*60}\n")
                break

    # =========================================
    # CONFRONTO AMBIENTALE
    # =========================================
    def check_comparison(self, frame):
        """Ogni ora confronta il frame corrente con quello precedente.
        
        Concentrato sull'ambiente (oggetti caduti, ostacoli) non sulla persona.
        Cattura cambiamenti graduali invisibili al diff frame-to-frame.
        """
        now = time.time()
        if now - self._last_comparison_time < self._comparison_interval:
            return

        self._last_comparison_time = now

        if self._comparison_frame is None:
            self._comparison_frame = self.capture.frame_to_base64(frame)
            self._comparison_frame_time = datetime.now().strftime("%H:%M")
            return

        current_b64 = self.capture.frame_to_base64(frame)
        now_str = datetime.now().strftime("%H:%M")

        prompt = (
            f"Ti mostro due immagini della stessa stanza. "
            f"La prima è delle ore {self._comparison_frame_time}, la seconda delle ore {now_str}. "
            f"Ignora la persona. Concentrati sull'AMBIENTE: "
            f"ci sono oggetti caduti, ostacoli nuovi, sedie spostate, "
            f"o qualsiasi cambiamento che potrebbe rappresentare un rischio? "
            f"Se non noti cambiamenti rilevanti, scrivi 'Ambiente invariato'. "
            f"Nota che l'inquadratura potrebbe essere cambiata."
        )

        images = [self._comparison_frame, current_b64]
        context = self._build_context()
        description = self.vlm.call_with_images(images, context, max_tokens=250, prompt_text=prompt)

        if description:
            obs = {
                "time": now_str,
                "timestamp": datetime.now().isoformat(),
                "hour": datetime.now().hour,
                "type": "confronto",
                "description": f"[CONFRONTO {self._comparison_frame_time}→{now_str}] {description}"
            }
            self.observations.append(obs)
            self._save()
            print(f"[{now_str}] [CMP] {description}")

        self._comparison_frame = current_b64
        self._comparison_frame_time = now_str