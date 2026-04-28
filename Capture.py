"""
Modulo di cattura frame.

Gestisce:
- Screen capture via mss
- Cattura finestra Xiaomi Home via Quartz (macOS)
- Change detection con soglia adattiva e mini-storia
- Burst capture per sequenze di frame
"""

import cv2
import time
import mss
import base64
import numpy as np
from collections import deque

# Cattura finestra specifica su macOS
try:
    import Quartz
    from Quartz import (CGWindowListCopyWindowInfo, kCGWindowListOptionAll,
                        kCGNullWindowID, CGWindowListCreateImage,
                        kCGWindowImageDefault, CGRectNull,
                        kCGWindowListOptionIncludingWindow)
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False


class CaptureManager:
    def __init__(self, monitor_area=None, use_window_capture=False):
        """
        Args:
            monitor_area: dict con top/left/width/height per screen capture
            use_window_capture: se True, prova a catturare la finestra Xiaomi Home
        """
        self.sct = mss.mss()
        self.monitor = monitor_area or {"top": 270, "left": 10, "width": 900, "height": 520}

        # Cattura finestra Xiaomi Home (macOS)
        self._xiaomi_window_id = None
        self._use_window_capture = use_window_capture
        if self._use_window_capture:
            self._xiaomi_window_id = self._find_xiaomi_window()
            if self._xiaomi_window_id:
                print(f"[INIT] Cattura finestra Xiaomi Home (window ID: {self._xiaomi_window_id})")
            else:
                print("[INIT] Finestra Xiaomi Home non trovata, uso screen capture")

        # Change detection
        self._prev_frame_gray = None
        self._diff_history = deque(maxlen=20)
        self._change_streak = 0
        self.last_diff = 0

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

    def capture_frame(self):
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

    # =========================================
    # CONVERSIONE E RIDIMENSIONAMENTO
    # =========================================
    def frame_to_base64(self, frame):
        """Ridimensiona e converte un frame in JPEG base64."""
        h, w = frame.shape[:2]
        max_size = 512
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return True

        diff = np.mean(np.abs(gray.astype(float) - self._prev_frame_gray.astype(float)))
        #print(f"[DIFF] {diff:.2f}", end=' | ')
        self._prev_frame_gray = gray
        self.last_diff = diff

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

    # =========================================
    # BURST CAPTURE
    # =========================================
    def capture_burst(self, n_frames=4, interval=0.5):
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
            frame = self.capture_frame()
            cv2.imshow("VLM Monitor - Preview (Q per chiudere)", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    @property
    def capture_mode(self):
        """Ritorna la modalità di cattura corrente."""
        return "Finestra Xiaomi Home" if self._xiaomi_window_id else "Screen capture"