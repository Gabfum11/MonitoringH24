# test_runner.py

"""
Esegue test clinici (TUG, STS) on-demand usando MediaPipe.
MediaPipe viene avviato solo per la durata del test, poi rilasciato.
"""

import time
import cv2
import mss
import numpy as np
from datetime import datetime

import PoseDetector as pm
from TugTest import TUGTest
from SitToStandTest import SitToStandTest
from Database_manager import DatabaseManager
from collections import deque


class TestRunner:
    def __init__(self, monitor_area=None, observations=None, save_callback=None):
        self.monitor_area = monitor_area or {"top": 270, "left": 10, "width": 900, "height": 520}
        self.db = DatabaseManager()
        self._running = False
        self._frame_times = deque(maxlen=60)
        self._observations = observations  # lista condivisa con il monitor
        self._save_callback = save_callback

    def _add_observation(self, description, obs_type='test'):
        """Aggiunge un risultato di test alle osservazioni del diario."""
        if self._observations is None:
            return
        now = datetime.now()
        self._observations.append({
            'hour': now.hour,
            'time': now.strftime("%H:%M"),
            'description': description,
            'type': obs_type
        })
        if self._save_callback:
            self._save_callback()

    def run_tug(self, timeout=120):
        """Esegue un test TUG. Ritorna il risultato o None."""
        print(f"\n[TEST] Avvio test TUG...")
        detector = pm.PoseDetector()
        detector.set_fps(30) #i 30 fps sono di default, ma in realtà li adatteremo dinamicamente in base al tempo reale che riusciamo a raggiungere
        detector.set_confidence_threshold(0.6)
        tug = TUGTest()
        sct = mss.mss()
        self._running = True
        self._frame_times.clear() 

        # Aspetta che la persona sia rilevata
        print("[TEST] In attesa della persona...")
        start_wait = time.time()
        hip_x, hip_y = 0, 0 

        while self._running and (time.time() - start_wait < 30):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)
            self._update_fps(detector)
            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                detector.detect_posture(None)
                hip_x = detector.hip_x
                hip_y = detector.hip_y
                print("[TEST] Persona rilevata, inizio TUG")
                break
            time.sleep(0.03)
        else:
            print("[TEST] Persona non rilevata, test annullato")
            self._running = False
            return None

        # Avvia il test
        tug.start(hip_x, hip_y)
        test_id = self.db.create_test_session(
            summary_date=datetime.now().strftime("%Y-%m-%d"),
            test_type="TUG",
            start_time=datetime.now().isoformat(),
            video_source="monitor_1" 
        )

        start_time = time.time()
        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)
            self._update_fps(detector)

            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                state = detector.detect_posture("TUG")
                knee_angle = detector.last_knee_angle
                movement = detector.last_movement
                phase = tug.update(state, detector.hip_x, detector.hip_y, movement, knee_angle)

                if phase == "FINISHED":
                    result = tug.get_result()
                    if result and test_id:
                        self.db.complete_test_session(test_id, datetime.now().isoformat())
                        self.db.save_tug_result(test_id, result)
                        t = result['total_time']
                        dist = result['total_distance_px']
                        speed = result['avg_speed_px_s']
                        print(f"[TEST] TUG completato: {t:.1f}s, {dist:.0f}px, {speed:.1f}px/s")
                        self._add_observation(
                            f"Test TUG completato: {t:.1f}s, "
                            f"distanza {dist:.0f}px, velocità media {speed:.1f}px/s"
                        )
                    self._running = False
                    return result

            time.sleep(0.03)

        print("[TEST] TUG timeout, test annullato")
        self._running = False
        return None

    def run_sts(self, timeout=120):
        """Esegue un test Sit-to-Stand (5 ripetizioni). Ritorna il risultato o None."""
        print(f"\n[TEST] Avvio test STS...")
        detector = pm.PoseDetector()
        detector.set_fps(30)
        detector.set_confidence_threshold(0.6)
        sts = SitToStandTest()
        sct = mss.mss()
        self._running = True
        self._frame_times.clear()

        # Aspetta persona
        print("[TEST] In attesa della persona...")
        start_wait = time.time()
        while self._running and (time.time() - start_wait < 30): #se entro 30 secondi non viene rilevata una persona, annulla il test
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            self._update_fps(detector)
            lmList = detector.findPosition(frame, draw=False)
            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                print("[TEST] Persona rilevata, inizio STS")
                break
            time.sleep(0.03)
        else:
            print("[TEST] Persona non rilevata, test annullato")
            self._running = False
            return None

        # Avvia il test
        sts.start()
        test_id = self.db.create_test_session(
            summary_date=datetime.now().strftime("%Y-%m-%d"),
            test_type="STS",
            start_time=datetime.now().isoformat(),
            video_source="monitor_1"
        )

        start_time = time.time()
        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)

            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                state = detector.detect_posture("STS")
                knee_angle = detector.last_knee_angle
                sts.update(state, knee_angle)

                if sts.reps >= 5:
                    result = sts.get_result()
                    if result and test_id:
                        self.db.complete_test_session(test_id, datetime.now().isoformat())
                        self.db.save_sts_result(test_id, result)
                        print(f"[TEST] STS completato: {result['reps_completed']} reps "
                              f"in {result['total_time']:.1f}s")
                        self._add_observation(
                            f"Test 5-Times Sit-to-Stand completato: {result['total_time']:.1f}s "
                            f"({result['reps_completed']} ripetizioni), "
                            f"tempo medio per ripetizione: {result['avg_rep_time']:.1f}s, "
                            f"angolo ginocchio medio: {result['avg_knee_angle']:.0f}°"
                        )
                    self._running = False
                    return result

            time.sleep(0.03)

        print("[TEST] STS timeout, test annullato")
        self._running = False
        return None

    def stop(self):
        """Ferma il test in corso."""
        self._running = False

    def _grab_frame(self, sct):
        """Cattura un frame dallo schermo."""
        sct_img = sct.grab(self.monitor_area)
        frame = np.array(sct_img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    def _update_fps(self, detector):
        """Calcola FPS reali e aggiorna il detector."""
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 30:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                real_fps = (len(self._frame_times) - 1) / span
                detector.set_fps(real_fps)