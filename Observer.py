"""
Modulo di osservazione.

Gestisce:
- Decisione se/come osservare (single, burst, burst_fast)
- Intervallo adattivo basato sul livello di movimento
- Filtro ridondanza per evitare osservazioni ripetitive
- Contesto conversazionale per il VLM
"""

import time
from datetime import datetime


class Observer:
    def __init__(self, capture_manager, vlm_client, observations, capture_interval=30):
        """
        Args:
            capture_manager: istanza di CaptureManager
            vlm_client: istanza di VLMClient
            observations: lista condivisa delle osservazioni (passata per riferimento)
            capture_interval: intervallo base in secondi
        """
        self.capture = capture_manager
        self.vlm = vlm_client
        self.observations = observations

        # Intervalli adattivi
        self._min_interval = capture_interval
        self._current_interval = capture_interval
        self._no_change_streak = 0
        self._prev_observation_time = 0
        self._prev_logged_interval = capture_interval

    @property
    def current_interval(self):
        return self._current_interval

    # =========================================
    # INTERVALLO ADATTIVO
    # =========================================
    def update_interval(self, scene_changed, last_diff):
        """Adatta l'intervallo in base al livello di movimento.
        
        Molto movimento (diff > 8): controlla ogni 10s
        Movimento moderato (diff > 5): controlla ogni 20s
        Scena stabile: intervallo cresce fino a 5min max
        """
        if scene_changed:
            self._no_change_streak = 0
            if last_diff > 10: 
                self._current_interval = 10
            elif last_diff > 5:
                self._current_interval = 20
            else:
                self._current_interval = self._min_interval
        else:
            self._no_change_streak += 1
            if self._no_change_streak % 5 == 0:
                self._current_interval = min(
                    self._current_interval * 2,
                    60 # max 1 minuto
                )

        # Log cambio intervallo
        if self._current_interval != self._prev_logged_interval:
            print(f"[INTERVAL] {self._prev_logged_interval}s → {self._current_interval}s "
                  f"({'movimento' if scene_changed else 'stabile'})")
            self._prev_logged_interval = self._current_interval

    # =========================================
    # DECISIONE: CHIAMARE IL VLM?
    # =========================================
    def should_observe(self, scene_changed, last_diff):
        """Decide se e come osservare.
        
        Ritorna:
            None: non osservare
            'single': frame singolo (scena stabile, check periodico)
            'burst': sequenza standard (4 frame ogni 0.5s)
            'burst_fast': sequenza rapida (5 frame ogni 0.3s)
        """
        now = time.time()
        time_since_last = now - self._prev_observation_time

        if scene_changed and time_since_last >= 15:
            if last_diff > 15: 
                return 'burst_fast'
            return 'burst'

        if time_since_last >= self._current_interval:
            return 'single'

        return None

    # =========================================
    # FILTRO RIDONDANZA
    # =========================================
    def _is_redundant(self, description):
        """Evita osservazioni ripetitive (stesso stato 3+ volte consecutive).
        
        Se le ultime 2 osservazioni e quella corrente contengono tutte
        la stessa keyword di stato, l'osservazione è ridondante.
        """
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

    # =========================================
    # CONTESTO CONVERSAZIONALE
    # =========================================
    def _build_context(self):
        """Contesto intelligente: ultime 3 osservazioni come riepilogo.
        
        Non costruisce una finta conversazione multi-turno,
        ma un riepilogo chiaro in un singolo messaggio.
        """
        if not self.observations:
            return None

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
    # OSSERVAZIONE
    # =========================================
    def observe(self, frame, mode='single'):
        """Analizza il frame (o una sequenza) e salva l'osservazione.
        
        Args:
            frame: frame corrente (usato per 'single')
            mode: 'single', 'burst', o 'burst_fast'
            
        Returns:
            str o None: la descrizione dell'osservazione, o None se skippata/errore
        """
        context = self._build_context()

        if mode == 'burst_fast':
            images = self.capture.capture_burst(n_frames=5, interval=0.3)
            description = self.vlm.call_with_images(images, context, max_tokens=250)
            obs_type = "sequenza_rapida"
            n_frames = 5
        elif mode == 'burst':
            images = self.capture.capture_burst(n_frames=4, interval=0.5)
            description = self.vlm.call_with_images(images, context, max_tokens=250)
            obs_type = "sequenza"
            n_frames = 4
        else:
            image_b64 = self.capture.frame_to_base64(frame)
            description = self.vlm.call_with_images(image_b64, context)
            obs_type = "singolo"
            n_frames = 1

        if description:
            # Filtro ridondanza
            if self._is_redundant(description):
                self._prev_observation_time = time.time()
                print(f"[{datetime.now().strftime('%H:%M')}] [SKIP] Stato invariato")
                return None

            obs = {
                "time": datetime.now().strftime("%H:%M"),
                "timestamp": datetime.now().isoformat(),
                "hour": datetime.now().hour,
                "type": obs_type,
                "description": description
            }
            self.observations.append(obs)
            self._prev_observation_time = time.time()

            # Tag con numero di frame per il log
            if mode == 'burst_fast':
                tag = f"FAS×{n_frames}"
            elif mode == 'burst':
                tag = f"SEQ×{n_frames}"
            else:
                tag = f"   ×{n_frames}"
            print(f"[{obs['time']}] [{tag}] {description}")

            return description
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] Nessuna risposta dal VLM")
            return None