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
    def __init__(self, monitor_area=None, observations=None, save_callback=None, vlm_client=None, output_dir="diari"):
        self.monitor_area = monitor_area or {"top": 270, "left": 10, "width": 900, "height": 520}
        self.db = DatabaseManager()
        self._running = False
        self._analysis_pending = False
        self._frame_times = deque(maxlen=60)
        self._observations = observations  # lista condivisa con il monitor
        self._save_callback = save_callback
        self._vlm = vlm_client
        self._output_dir = output_dir

    @property
    def is_busy(self):
        """True se è in corso un test o la sua analisi VLM post-test."""
        return self._running or self._analysis_pending

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

        # Attende la posizione seduta stabile e raccoglie baseline anca
        print("[TEST] In attesa che la persona si sieda...")
        start_wait = time.time()
        hip_x, hip_y = 0, 0
        seated_since = None
        SEAT_HOLD = 1.0
        HIP_LIFT_RATIO = TUGTest.HIP_LIFT_RATIO
        baseline_hip_y = None
        baseline_torso = None
        hip_lift_threshold = None
        _hip_y_samples = []
        _torso_samples = []

        while self._running and (time.time() - start_wait < 60):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)
            self._update_fps(detector)
            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                state = detector.detect_posture(None)
                if state == "SITTING" and detector.last_knee_angle > 0:
                    if seated_since is None:
                        seated_since = time.time()
                    else:
                        _hip_y_samples.append(detector.hip_y)
                        if detector.torso_length > 0:
                            _torso_samples.append(detector.torso_length)
                        if (time.time() - seated_since >= SEAT_HOLD
                                and baseline_hip_y is None
                                and len(_hip_y_samples) >= 10):
                            baseline_hip_y = sum(_hip_y_samples) / len(_hip_y_samples)
                            if _torso_samples:
                                baseline_torso = sum(_torso_samples) / len(_torso_samples)
                                hip_lift_threshold = baseline_torso * HIP_LIFT_RATIO
                            else:
                                baseline_torso = 0
                                hip_lift_threshold = 15.0
                            hip_x = detector.hip_x
                            hip_y = detector.hip_y
                            print(f"[TUG] Baseline hip_y={baseline_hip_y:.1f}, "
                                  f"busto={baseline_torso:.1f}px, soglia={hip_lift_threshold:.1f}px")
                            print("[TEST] Persona seduta stabile, inizio TUG")
                            break
                else:
                    seated_since = None
                    _hip_y_samples.clear()
                    _torso_samples.clear()
            time.sleep(0.03)
        else:
            print("[TEST] Posizione seduta non rilevata, test annullato")
            self._running = False
            return None

        # Avvia il test con baseline pre-calcolata
        tug.start(hip_x, hip_y)
        tug.baseline_hip_y = baseline_hip_y
        tug.baseline_torso_length = baseline_torso
        tug.hip_lift_threshold_px = hip_lift_threshold
        now_dt = datetime.now()
        test_timestamp = now_dt.isoformat()
        test_dir_ts = now_dt.strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        person_lost_since = None
        MAX_CONTINUOUS_LOSS = 2.0  # secondi consecutivi senza persona in inquadratura
        last_landmark_log = 0  # timestamp dell'ultimo log periodico sui landmark

        # Buffer per il campionamento periodico a fini di analisi VLM
        ANALYSIS_CAPTURE_INTERVAL = 0.3  # secondi
        ANALYSIS_TARGET_FRAMES = 15
        analysis_buffer = []
        last_analysis_capture = 0

        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=True)
            lmList = detector.findPosition(frame, draw=False)
            self._update_fps(detector)

            person_in_frame = len(lmList) > 0

            # Log periodico (~1Hz) sul numero di landmark visibili
            if time.time() - last_landmark_log >= 1.0:
                visible_landmarks = sum(1 for lm in lmList if lm[3] > 0.5)
                print(f"[TUG-TRACK] fase={tug.phase} landmark visibili={visible_landmarks}/{len(lmList)}")
                last_landmark_log = time.time()

            if person_in_frame:
                person_lost_since = None

                state = detector.detect_posture("TUG")
                knee_angle = detector.last_knee_angle
                movement = detector.last_movement

                phase = tug.update(state, detector.hip_x, detector.hip_y, movement, knee_angle, detector.torso_length)

                # Campionamento periodico per l'analisi VLM — frame pulito (senza overlay testo)
                if (tug.start_time is not None
                        and time.time() - last_analysis_capture >= ANALYSIS_CAPTURE_INTERVAL):
                    analysis_buffer.append(self._frame_to_base64(frame))
                    last_analysis_capture = time.time()

            if person_in_frame:
                if phase == "FINISHED":
                    result = tug.get_result()
                    test_id = None
                    if result:
                        test_id = self.db.save_tug_result(result, test_timestamp)
                        t = result['total_time']
                        obs_text = f"Test TUG completato: {t:.1f}s"
                        print(f"[TEST] TUG: {obs_text}")
                        self._add_observation(obs_text)
                    self._running = False
                    analysis_frames = analysis_buffer
                    if len(analysis_buffer) > ANALYSIS_TARGET_FRAMES:
                        step = len(analysis_buffer) / ANALYSIS_TARGET_FRAMES
                        analysis_frames = [analysis_buffer[int(i * step)] for i in range(ANALYSIS_TARGET_FRAMES)]
                    print(f"[TUG] Buffer analisi: {len(analysis_buffer)} catturati → {len(analysis_frames)} inviati al VLM")
                    self._save_frames(analysis_frames, "TUG", "analisi", test_dir_ts)
                    return result, analysis_frames, test_id

            else:
                if person_lost_since is None:
                    person_lost_since = time.time()
                elif time.time() - person_lost_since > MAX_CONTINUOUS_LOSS:
                    print(f"[TEST] Persona non visibile da più di {MAX_CONTINUOUS_LOSS}s in fase {tug.phase}, test annullato")
                    self._add_observation(
                        f"Test TUG annullato: persona non visibile per più di {MAX_CONTINUOUS_LOSS:.0f} secondi durante la fase {tug.phase}"
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

        # Attende la posizione seduta stabile e raccoglie baseline anca
        print("[TEST] In attesa che la persona si sieda...")
        start_wait = time.time()
        seated_since = None
        SEAT_HOLD = 1.0
        HIP_LIFT_RATIO = 0.12
        baseline_hip_y = None
        baseline_torso = None
        hip_lift_threshold = None
        _hip_y_samples = []
        _torso_samples = []
        while self._running and (time.time() - start_wait < 60):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            self._update_fps(detector)
            lmList = detector.findPosition(frame, draw=False)
            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                state = detector.detect_posture(None)
                if state == "SITTING" and detector.last_knee_angle > 0:
                    if seated_since is None:
                        seated_since = time.time()
                    else:
                        _hip_y_samples.append(detector.hip_y)
                        if detector.torso_length > 0:
                            _torso_samples.append(detector.torso_length)
                        if (time.time() - seated_since >= SEAT_HOLD
                                and baseline_hip_y is None
                                and len(_hip_y_samples) >= 10
                                and len(_torso_samples) >= 5):
                            baseline_hip_y = sum(_hip_y_samples) / len(_hip_y_samples)
                            baseline_torso = sum(_torso_samples) / len(_torso_samples)
                            hip_lift_threshold = baseline_torso * HIP_LIFT_RATIO
                            print(f"[STS] Baseline hip_y={baseline_hip_y:.1f}, "
                                  f"busto={baseline_torso:.1f}px, soglia={hip_lift_threshold:.1f}px")
                            print("[TEST] Persona seduta stabile, inizio STS")
                            break
                else:
                    seated_since = None
                    _hip_y_samples.clear()
                    _torso_samples.clear()
            time.sleep(0.03)
        else:
            print("[TEST] Posizione seduta non rilevata, test annullato")
            self._running = False
            return None

        # Avvia il test
        sts.start()
        now_dt = datetime.now()
        test_timestamp = now_dt.isoformat()
        test_dir_ts = now_dt.strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        tracking_lost_since = None
        person_lost_since = None
        MAX_CONTINUOUS_LOSS_STS = 2.0
        last_rep_count = 0
        last_rep_time = time.time()
        rep_start_time = time.time()
        rep_times = []
        inactivity_timeout = 20
        ANALYSIS_CAPTURE_INTERVAL = 0.3
        ANALYSIS_TARGET_FRAMES = 15
        analysis_buffer = []
        last_analysis_capture = 0

        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=True)
            lmList = detector.findPosition(frame, draw=False)

            person_in_frame = len(lmList) > 0

            if tracking_lost_since is not None and len(lmList) > 0:
                tracking_lost_since = None
            elif tracking_lost_since is None and len(lmList) == 0:
                tracking_lost_since = time.time()

            if person_in_frame:
                person_lost_since = None
                state = detector.detect_posture("STS")
                knee_angle = detector.last_knee_angle

                # Rileva alzata tramite soglia anca, come in TUG
                if sts.start_time is None and baseline_hip_y is not None:
                    if baseline_hip_y - detector.hip_y > hip_lift_threshold:
                        sts.start_time = time.time()
                        rep_start_time = sts.start_time
                        delta = baseline_hip_y - detector.hip_y
                        print(f"[STS] Hip lift rilevato, timer avviato (Δy={delta:.1f}px)")

                sts.update(state, knee_angle)

                # Campionamento periodico per l'analisi VLM — frame pulito
                if (sts.start_time is not None
                        and time.time() - last_analysis_capture >= ANALYSIS_CAPTURE_INTERVAL):
                    analysis_buffer.append(self._frame_to_base64(frame))
                    last_analysis_capture = time.time()

                if sts.reps > last_rep_count:
                    rep_times.append(time.time() - rep_start_time)
                    rep_start_time = time.time()
                    last_rep_count = sts.reps
                    last_rep_time = time.time()
                elif sts.reps < 5 and time.time() - last_rep_time > inactivity_timeout:
                    print(f"[TEST] STS interrotto per inattività ({sts.reps}/5 rep)")
                    self._add_observation(f"Test STS interrotto: {sts.reps}/5 ripetizioni completate")
                    partial = {'total_time': round(time.time() - start_time, 2), 'reps_completed': sts.reps}
                    test_id = self.db.save_sts_result(partial, test_timestamp, completed=0)
                    if test_id and rep_times:
                        self.db.save_sts_reps(test_id, rep_times)
                    self._running = False
                    return None

            if person_in_frame and sts.reps >= 5 and not sts.test_active:
                result = sts.get_result()
                test_id = None
                if result:
                    test_id = self.db.save_sts_result(result, test_timestamp)
                    if test_id and rep_times:
                        self.db.save_sts_reps(test_id, rep_times)
                    t = result['total_time']
                    obs_text = f"Test STS completato: {t:.1f}s (5 ripetizioni)"
                    print(f"[TEST] STS: {obs_text}")
                    self._add_observation(obs_text)
                self._running = False
                analysis_frames = analysis_buffer
                if len(analysis_buffer) > ANALYSIS_TARGET_FRAMES:
                    step = len(analysis_buffer) / ANALYSIS_TARGET_FRAMES
                    analysis_frames = [analysis_buffer[int(i * step)] for i in range(ANALYSIS_TARGET_FRAMES)]
                print(f"[STS] Buffer analisi: {len(analysis_buffer)} catturati → {len(analysis_frames)} inviati al VLM")
                self._save_frames(analysis_frames, "STS", "analisi", test_dir_ts)
                return result, analysis_frames, test_id

            if not person_in_frame:
                if person_lost_since is None:
                    person_lost_since = time.time()
                elif time.time() - person_lost_since > MAX_CONTINUOUS_LOSS_STS:
                    print(f"[TEST] Persona non visibile da più di {MAX_CONTINUOUS_LOSS_STS}s, test STS annullato (rep {sts.reps}/5)")
                    self._add_observation(
                        f"Test STS annullato: persona non visibile per più di {MAX_CONTINUOUS_LOSS_STS:.0f} secondi (completate {sts.reps}/5 ripetizioni)"
                    )
                    partial = {'total_time': round(time.time() - start_time, 2), 'reps_completed': sts.reps}
                    test_id = self.db.save_sts_result(partial, test_timestamp, completed=0)
                    if test_id and rep_times:
                        self.db.save_sts_reps(test_id, rep_times)
                    self._running = False
                    return None
            else:
                person_lost_since = None

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

    def _frame_to_base64(self, frame):
        """Converte un frame OpenCV in base64."""
        import base64
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def _save_frames(self, frames_b64: list, test_type: str, label: str, test_ts: str = None):
        """Salva i frame base64 come JPEG in una cartella dedicata al singolo test.

        Args:
            frames_b64: lista di frame in base64 da salvare.
            test_type: tipo del test (es. "TUG", "STS").
            label: etichetta che descrive il contenuto dei frame (es. "alzata", "fasi").
            test_ts: timestamp condiviso del test; se omesso ne viene generato uno nuovo.
        """
        import base64, os
        if test_ts is None:
            test_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(self._output_dir, "test_frames", f"{test_ts}_{test_type}")
        os.makedirs(folder, exist_ok=True)
        for i, b64 in enumerate(frames_b64):
            path = os.path.join(folder, f"{label}_{i+1}.jpg")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
        print(f"[TEST] Salvati {len(frames_b64)} frame in {folder}/")

    def _analyse_sts_quality(self, analysis_frames: list):
        """Analisi qualitativa VLM sui frame del test STS.

        Ritorna: (testo_referto, durata_secondi)
        """
        prompt = (
            f"Ti mostro una sequenza di {len(analysis_frames)} frame in ordine temporale, "
            "catturati durante l'esecuzione del test Sit-to-Stand (5 ripetizioni). "
            "Valuta complessivamente il movimento e rispondi in 2-3 frasi: "
            "la persona ha usato i braccioli o le mani per alzarsi? "
            "Il movimento è fluido o si notano difficoltà e instabilità?"
        )
        self._analysis_pending = True
        try:
            t0 = time.perf_counter()
            text = self._vlm.call_with_images(
                analysis_frames, prompt_text=prompt, max_tokens=100, category="analisi_test_sts"
            )
            return text, round(time.perf_counter() - t0, 2)
        finally:
            self._analysis_pending = False

    def _analyse_tug(self, frames: list):
        """Analisi VLM unificata sui frame del test TUG (alzata + andatura + svolta + ritorno).

        Ritorna: (testo_referto, durata_secondi)
        """
        prompt = (
            f"Ti mostro una sequenza di {len(frames)} frame in ordine temporale, "
            "catturati durante l'esecuzione del test Timed Up and Go. "
            "I frame coprono in ordine l'intera esecuzione del test: l'alzata dalla sedia, "
            "la camminata, la svolta, il ritorno e la riseduta. "
            "Valuta complessivamente l'esecuzione e rispondi in 3-4 frasi:\n"
            "1) la persona ha usato i braccioli o le mani per alzarsi? il movimento di alzata è fluido o faticoso?\n"
            "2) l'andatura è stabile e simmetrica? la svolta appare fluida? si notano esitazioni, "
            "oscillazioni laterali o appoggi compensatori?\n"
            "3) la persona è stata sempre ben visibile e completamente inquadrata durante il test, "
            "oppure alcune parti del corpo sono uscite dall'inquadratura o sono state coperte?"
        )
        self._analysis_pending = True
        try:
            t0 = time.perf_counter()
            text = self._vlm.call_with_images(
                frames, prompt_text=prompt, max_tokens=200, category="analisi_test_tug"
            )
            return text, round(time.perf_counter() - t0, 2)
        finally:
            self._analysis_pending = False

    def _update_fps(self, detector):
        """Calcola FPS reali e aggiorna il detector."""
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 30:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                real_fps = (len(self._frame_times) - 1) / span
                detector.set_fps(real_fps)