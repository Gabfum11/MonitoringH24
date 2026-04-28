"""
Modulo di rilevamento assenza e confronto ambientale.

Gestisce:
- Tracciamento assenza prolungata con alert
- Osservazioni di confronto ambientale periodiche
"""

import time
from datetime import datetime


class AbsenceTracker:
    def __init__(self, capture_manager, vlm_client, observations, save_callback,
                 comparison_interval=3600):
        """
        Args:
            capture_manager: istanza di CaptureManager
            vlm_client: istanza di VLMClient
            observations: lista condivisa delle osservazioni
            save_callback: funzione da chiamare per salvare i dati
            comparison_interval: secondi tra confronti ambientali (default: 1 ora)
        """
        self.capture = capture_manager
        self.vlm = vlm_client
        self.observations = observations
        self._save = save_callback

        # Assenza
        self._consecutive_absence = 0
        self._absence_alerted = False
        self._absence_start_time = 0

        # Confronto ambientale
        self._last_comparison_time = time.time()
        self._comparison_interval = comparison_interval
        self._comparison_frame = None
        self._comparison_frame_time = None

    def reset(self):
        """Reset giornaliero dello stato."""
        self._consecutive_absence = 0
        self._absence_alerted = False
        self._comparison_frame = None

    # =========================================
    # RILEVAMENTO ASSENZA PROLUNGATA
    # =========================================
    def track_absence(self, description):
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
        else:
            self._consecutive_absence = 0
            self._absence_alerted = False

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
    # OSSERVAZIONE DI CONFRONTO AMBIENTALE
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
        # Contesto dalle ultime osservazioni
        context = None
        if self.observations:
            recent = self.observations[-3:]
            summary = "Osservazioni precedenti:\n"
            for obs in recent:
                summary += f"- Ore {obs['time']}: {obs['description']}\n"
            context = [{"role": "user", "content": summary + "\nOra confronta i due frame."}]

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