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
        tracking_lost_since = None
        tracking_lost_phases = []

        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)
            self._update_fps(detector)

            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                if tracking_lost_since is not None:
                    lost_duration = time.time() - tracking_lost_since
                    tracking_lost_phases.append((tug.phase, round(lost_duration, 1)))
                    print(f"[TEST] Tracking recuperato dopo {lost_duration:.1f}s (fase: {tug.phase})")
                    tracking_lost_since = None

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
                        if t < 12:
                            giudizio = "mobilità nella norma"
                        elif t < 20:
                            giudizio = "rischio moderato di caduta"
                        else:
                            giudizio = "rischio elevato di caduta"

                        obs_text = f"Test TUG completato: {t:.1f}s — {giudizio}"
                        if tracking_lost_phases:
                            fasi = ", ".join(f"{fase} ({dur}s)" for fase, dur in tracking_lost_phases)
                            obs_text += f" [tracking perso durante: {fasi}]"

                        print(f"[TEST] TUG completato: {t:.1f}s ({giudizio})")
                        if tracking_lost_phases:
                            print(f"[TEST] Perdite tracking: {tracking_lost_phases}")
                        self._add_observation(obs_text)
                    self._running = False
                    return result

            else:
                if tracking_lost_since is None:
                    tracking_lost_since = time.time()
                elif time.time() - tracking_lost_since > 2.0:
                    print(f"[TEST] Tracking perso da più di 2s in fase {tug.phase}, test annullato")
                    self._add_observation(
                        f"Test TUG annullato: tracking perso per più di 2 secondi durante la fase {tug.phase}"
                    )
                    self._running = False
                    return None

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
        tracking_lost_since = None
        last_rep_count = 0
        last_rep_time = time.time()
        inactivity_timeout = 20

        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)

            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                tracking_lost_since = None
                state = detector.detect_posture("STS")
                knee_angle = detector.last_knee_angle
                sts.update(state, knee_angle)

                if sts.reps > last_rep_count:
                    last_rep_count = sts.reps
                    last_rep_time = time.time()
                elif sts.reps < 5 and time.time() - last_rep_time > inactivity_timeout:
                    print(f"[TEST] STS interrotto per inattività ({sts.reps}/5 rep)")
                    self._add_observation(f"Test STS interrotto: {sts.reps}/5 ripetizioni completate")
                    self._running = False
                    return None

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

            else:
                if tracking_lost_since is None:
                    tracking_lost_since = time.time()
                elif time.time() - tracking_lost_since > 1.0:
                    print(f"[TEST] Tracking perso da più di 1s, test STS annullato (rep {sts.reps}/5)")
                    self._add_observation(
                        f"Test STS annullato: persona non visibile per più di 1 secondo (completate {sts.reps}/5 ripetizioni)"
                    )
                    self._running = False
                    return None

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