"""
Modulo di osservazione.

Gestisce:
- Decisione se/come osservare (single, sequenza, sequenza_rapida)
- Intervallo adattivo basato sul livello di movimento
- Filtro ridondanza per evitare osservazioni ripetitive
- Contesto conversazionale per il VLM
- Rilevamento assenza prolungata con alert
- Osservazioni di confronto ambientale periodiche (sperimentale)
"""

import re
import time
from datetime import datetime


class Observer:
    def __init__(self, capture_manager, vlm_client, observations,
                 save_callback, capture_interval=30, comparison_interval=1800,
                 alert_callback=None, output_dir="diari"):
        """
        Args:
            capture_manager: istanza di CaptureManager
            vlm_client: istanza di VLMClient
            observations: lista condivisa delle osservazioni (passata per riferimento)
            save_callback: funzione da chiamare per salvare i dati su disco
            capture_interval: intervallo base in secondi
            comparison_interval: secondi tra confronti ambientali (default: 30 minuti)
            alert_callback: funzione opzionale chiamata con il testo dell'alert
            output_dir: cartella radice dei diari, usata per salvare i frame degli alert
        """
        self.capture = capture_manager
        self.vlm = vlm_client
        self.observations = observations
        self._save = save_callback
        self._alert_callback = alert_callback
        self._output_dir = output_dir

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

        if scene_changed:
            if last_diff > 20 or change_streak >= 3:
                # Evento significativo o azione sostenuta: cooldown minimo per non sovrapporre il buffer
                if time_since_last >= 30:
                    if last_diff > 6 and change_streak >= 3:
                        return 'sequenza_rapida'
                    return 'sequenza'
            else:
                # Spike lieve (10-20): micro-movimento vicino camera, cooldown lungo
                if time_since_last >= 60:
                    return 'sequenza'

        if time_since_last >= self._current_interval:
            return 'single'

        return None

    # =========================================
    # CONTESTO CONVERSAZIONALE
    # =========================================
    def _build_context(self):
        """Contesto intelligente: ultime 5 osservazioni come riepilogo e il riassunto dell'ultima ora.

        Per evitare un effetto "priming" che porti il VLM a ripetere alert già emessi,
        i prefissi e i marcatori di alert vengono rimossi dalle descrizioni nel contesto.
        """
        if not self.observations:
            return None

        summary = ""
        if self._last_hourly_text:
            summary += f"Riepilogo dell'ultima ora: {self._last_hourly_text}\n\n"

        recent = self.observations[-5:]
        summary += "Osservazioni recenti:\n"
        for obs in recent:
            description = obs['description']
            description = re.sub(r'^ALERT\s*\([^)]*\):\s*', '', description)
            description = re.sub(r'\[ALERT:\s*[^\]]*\]', '', description).strip()
            summary += f"- Ore {obs['time']}: {description}\n"

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
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] Nessuna risposta dal VLM")
            return False

    # =========================================
    # RILEVAMENTO ASSENZA PROLUNGATA
    # =========================================
    def _track_absence(self, description):
        """Traccia osservazioni consecutive senza persona visibile.

        Se la persona non è visibile per 60+ minuti durante il giorno (6-22),
        genera un alert e lo aggiunge alle osservazioni.
        Le ore notturne non contribuiscono al conteggio (è normale che la
        persona stia dormendo in un'altra stanza).
        """
        hour = datetime.now().hour
        is_daytime = 6 <= hour < 22

        # Reset del conteggio nelle ore notturne
        if not is_daytime:
            self._consecutive_absence = 0
            self._absence_alerted = False
            return

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
        else:
            self._consecutive_absence = 0
            self._absence_alerted = False

        # Calcolo basato sul tempo reale
        if self._consecutive_absence > 0:
            minutes_absent = (time.time() - self._absence_start_time) / 60
        else:
            minutes_absent = 0

        if minutes_absent >= 60 and not self._absence_alerted:
            alert_obs = {
                "time": datetime.now().strftime("%H:%M"),
                "timestamp": datetime.now().isoformat(),
                "hour": datetime.now().hour,
                "type": "assenza",
                "description": (f"ASSENZA: La persona non è visibile da circa "
                               f"{minutes_absent:.0f} minuti durante le ore diurne.")
            }
            self.observations.append(alert_obs)
            self._save()
            self._absence_alerted = True
            print(f"\n{'!'*60}")
            print(f"[{alert_obs['time']}] {alert_obs['description']}")
            print(f"{'!'*60}\n")
            if self._alert_callback:
                self._alert_callback(alert_obs['description'])

    # =========================================
    # CONFRONTO AMBIENTALE (sperimentale)
    # =========================================
    def check_comparison(self, frame):
        """Ogni intervallo confronta il frame corrente con quello precedente.

        Concentrato sull'ambiente (oggetti caduti, ostacoli, arredi spostati),
        non sulla persona. Il prompt è formulato in modo direttivo con esempi
        few-shot per ridurre il default "Ambiente invariato" emesso dal VLM.
        Il contesto conversazionale viene volutamente omesso per evitare il
        bias persona-centrico delle osservazioni recenti.
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
            f"Confronta queste due immagini DELLA STESSA STANZA scattate a "
            f"{self._comparison_frame_time} e a {now_str}.\n"
            f"Il tuo unico compito è trovare DIFFERENZE NEGLI OGGETTI o "
            f"NELL'ARREDAMENTO.\n"
            f"Esempi di cosa devi segnalare:\n"
            f"- una sedia è in una posizione diversa\n"
            f"- c'è un oggetto sul pavimento che prima non c'era\n"
            f"- una porta è aperta dove prima era chiusa\n"
            f"- un mobile è stato spostato\n"
            f"NON descrivere persone. NON dire \"invariato\" se non sei sicuro.\n"
            f"Elenca ogni differenza che vedi, anche piccola."
        )

        images = [self._comparison_frame, current_b64]
        description = self.vlm.call_with_images(
            images, context_messages=None, max_tokens=250, prompt_text=prompt,
            category="confronto"
        )

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