"""
Modulo di cattura frame.

Gestisce:
- Screen capture via mss
- Change detection con soglia adattiva e mini-storia
- Burst capture per sequenze di frame
"""

import cv2
import time
import mss
import base64
import numpy as np
import subprocess
from collections import deque


class CaptureManager:
    def __init__(self, monitor_area=None):
        """
        Args:
            monitor_area: dict con top/left/width/height per screen capture
        """
        self.sct = mss.mss()
        self.monitor = monitor_area or {"top": 270, "left": 10, "width": 900, "height": 520}
        self.frame_buffer = deque(maxlen=10)  # Buffer per burst capture

        # Change detection
        self._prev_frame_gray = None
        self._diff_history = deque(maxlen=10) 
        self._change_streak = 0
        self.last_diff = 0

    # =========================================
    # CATTURA FRAME
    # =========================================
    def capture_frame(self):
        """Cattura un frame dallo schermo."""
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    # =========================================
    # CONVERSIONE E RIDIMENSIONAMENTO
    # =========================================
    def frame_to_base64(self, frame):
        """Ridimensiona e converte un frame in JPEG base64."""
        h, w = frame.shape[:2]
        max_size = 768
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(jpg.tobytes()).decode('utf-8')

    # =========================================
    # CHANGE DETECTION
    # =========================================
    def scene_changed(self, frame):
        """Confronta il frame corrente con il precedente.
        
        Usa soglia adattiva (media + 2σ dei diff recenti) e mini-storia
        (2 frame consecutivi sopra soglia) per filtrare falsi positivi.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        self.frame_buffer.append(frame.copy())

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return True

        diff = np.mean(np.abs(gray.astype(float) - self._prev_frame_gray.astype(float)))
        self._prev_frame_gray = gray
        self.last_diff = diff
        if diff > 70:
            self._change_streak = 0
            print(f"[ROTAZIONE] Movimento camera rilevato (Diff: {diff:.2f}), ignoro...")
            return False

        # Soglia adattiva
        self._diff_history.append(diff)
        if len(self._diff_history) >= 5:
            mean_diff = np.mean(list(self._diff_history))
            std_diff = np.std(list(self._diff_history))
            threshold = max(2.5, mean_diff + 1.0 * std_diff)
        else:
            threshold = 2.5

        # Mini-storia: 2 frame consecutivi sopra soglia
        if diff > threshold:
            self._change_streak += 1
        else:
            self._change_streak = 0

        print(f"[DIFF] diff={diff:.2f} threshold={threshold:.2f} streak={self._change_streak} changed={self._change_streak >= 2}")
        return self._change_streak >= 2
    
   
    # =========================================
    # BURST CAPTURE
    # =========================================
    def capture_sequenza(self, n_frames=3, interval=2):
        """Cattura una sequenza rapida di frame per analizzare un'azione.
        
        Args:
            n_frames: quanti frame catturare
            interval: secondi tra i frame
        """
        frames = []
        for i in range(n_frames):
            frame = self.capture_frame()
            frames.append(self.frame_to_base64(frame))
            if i < n_frames - 1:
                time.sleep(interval)
        return frames

    def get_strategic_frames(self):
        """Recupera 4 frame distribuiti sui 20 secondi di buffer."""
        # Trasforma la deque in lista per accedere agli indici
        buffer_list = list(self.frame_buffer)
        n = len(buffer_list)
        
        # Se il buffer è quasi vuoto, prendi quello che c'è
        if n < 4:
            return [self.frame_to_base64(f) for f in buffer_list]
            
        #0 -> t-20s, n//3 -> t-13s, (2n)//3 -> t-7s, n-1 -> T-2s
        indices = [0, n // 3, (2 * n) // 3, n - 1]
        
        return [self.frame_to_base64(buffer_list[i]) for i in indices]

    # =========================================
    # ZOOM AUTOMATICO
    # =========================================
    def _click_center(self):
        cx = self.monitor['left'] + self.monitor['width'] // 2
        cy = self.monitor['top'] + self.monitor['height'] // 2
        subprocess.run(
            ['osascript', '-e',
             f'tell application "System Events" to double click at {{{cx}, {cy}}}'],
            capture_output=True
        )

    def zoom_in(self):
        self._click_center()
        time.sleep(2)
        print("[ZOOM] Zoom avanti")

    def zoom_out(self):
        self._click_center()
        time.sleep(2)
        print("[ZOOM] Zoom indietro")

    # =========================================
    # ANTEPRIMA
    # =========================================
    def preview(self):
        """Mostra una finestra con l'area catturata. Premi Q per chiudere."""
        print(f"[PREVIEW] Screen capture - Area: top={self.monitor['top']} left={self.monitor['left']} "
              f"{self.monitor['width']}x{self.monitor['height']}")
        print("Premi Q per chiudere l'anteprima e avviare il monitoraggio\n")

        cv2.namedWindow("VLM Monitor - Preview (Q per chiudere)", cv2.WINDOW_NORMAL)
        cv2.moveWindow("VLM Monitor - Preview (Q per chiudere)", 1000, 0)

        while True:
            frame = self.capture_frame()
            cv2.imshow("VLM Monitor - Preview (Q per chiudere)", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    @property
    def capture_mode(self):
        return "Screen capture"