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
        self._frame_times = deque(maxlen=60)
        self._observations = observations  # lista condivisa con il monitor
        self._save_callback = save_callback
        self._vlm = vlm_client
        self._output_dir = output_dir

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
        tug_frames = []
        standup_frames = []
        _standup_captured = False
        _tug_frame_buffer = deque(maxlen=5)  # ~0.15s a 30fps

        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)
            self._update_fps(detector)
            _tug_frame_buffer.append(frame)

            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                if tracking_lost_since is not None:
                    lost_duration = time.time() - tracking_lost_since
                    tracking_lost_phases.append((tug.phase, round(lost_duration, 1)))
                    print(f"[TEST] Tracking recuperato dopo {lost_duration:.1f}s (fase: {tug.phase})")
                    tracking_lost_since = None

                state = detector.detect_posture("TUG")
                knee_angle = detector.last_knee_angle
                movement = detector.last_movement

                # Al primo non-SITTING: cattura il frame di ~0.15s fa (persona in spinta)
                if tug.phase == "SIT_TO_STAND" and state != "SITTING" and not _standup_captured:
                    f = _tug_frame_buffer[0] if len(_tug_frame_buffer) == _tug_frame_buffer.maxlen else frame
                    standup_frames.append(self._frame_to_base64(f))
                    _standup_captured = True
                    print(f"[TUG-STANDUP] Catturato frame alzata dal buffer (stato: {state})")

                phase = tug.update(state, detector.hip_x, detector.hip_y, movement, knee_angle)

                if tug.phase_just_changed:
                    tug_frames.append(self._frame_to_base64(frame))
                    print(f"[TUG-FRAME] Catturato frame per fase {tug.phase}, totale: {len(tug_frames)}")

                if phase == "FINISHED":
                    result = tug.get_result()
                    if result and test_id:
                        # Legge il test precedente prima di salvare quello nuovo
                        today = datetime.now().strftime("%Y-%m-%d")
                        prev_results = self.db.get_tug_results("2000-01-01", today)
                        prev = prev_results[-1] if prev_results else None

                        self.db.complete_test_session(test_id, datetime.now().isoformat())
                        self.db.save_tug_result(test_id, result)

                        t = result['total_time']
                        obs_text = f"Test TUG completato: {t:.1f}s"

                        # Trend velocità normalizzata rispetto al test precedente
                        if prev and prev['avg_speed_px_s'] and result['avg_speed_px_s']:
                            delta_pct = ((result['avg_speed_px_s'] - prev['avg_speed_px_s']) / prev['avg_speed_px_s']) * 100
                            delta_t = t - prev['total_time']
                            if abs(delta_pct) > 5:
                                direzione = "miglioramento" if delta_pct > 0 else "peggioramento"
                                if abs(delta_t) < 1.0:
                                    tempo_note = ", nonostante tempo simile"
                                else:
                                    tempo_note = ""
                                obs_text += f" — {direzione} del {abs(delta_pct):.0f}% nella velocità di cammino{tempo_note}"
                            else:
                                obs_text += " — prestazione simile al test precedente"

                        if tracking_lost_phases:
                            fasi = ", ".join(f"{fase} ({dur}s)" for fase, dur in tracking_lost_phases)
                            obs_text += f" [tracking perso durante: {fasi}]"

                        print(f"[TEST] TUG: {obs_text}")
                        self._add_observation(obs_text)
                    self._running = False
                    if standup_frames:
                        self._save_frames(standup_frames, "TUG", "alzata")
                    if tug_frames:
                        self._save_frames(tug_frames, "TUG", "fasi")
                    return result, standup_frames, tug_frames, test_id

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
        transition_frames = []
        _frame_buffer = deque(maxlen=5)  # ~0.15s a 30fps

        while self._running and (time.time() - start_time < timeout):
            frame = self._grab_frame(sct)
            frame = detector.findPose(frame, draw=False)
            lmList = detector.findPosition(frame, draw=False)
            _frame_buffer.append(frame)

            if detector.tracking_quality >= 0.75 and len(lmList) > 0:
                tracking_lost_since = None
                state = detector.detect_posture("STS")
                knee_angle = detector.last_knee_angle
                sts.update(state, knee_angle)

                if sts.standup_just_occurred:
                    # Usa il frame più vecchio del buffer (~0.5s fa) per catturare la spinta
                    f = _frame_buffer[0] if len(_frame_buffer) == _frame_buffer.maxlen else frame
                    transition_frames.append(self._frame_to_base64(f))
                if sts.transition_just_occurred:
                    transition_frames.append(self._frame_to_base64(frame))

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
                        today = datetime.now().strftime("%Y-%m-%d")
                        prev_results = self.db.get_sts_results("2000-01-01", today)
                        prev = prev_results[-1] if prev_results else None

                        self.db.complete_test_session(test_id, datetime.now().isoformat())
                        self.db.save_sts_result(test_id, result)

                        t = result['total_time']
                        obs_text = f"Test STS completato: {t:.1f}s (5 ripetizioni)"

                        if prev and prev['total_time']:
                            delta = t - prev['total_time']
                            if delta > 1:
                                obs_text += f" — peggioramento (+{delta:.1f}s rispetto al test precedente)"
                            elif delta < -1:
                                obs_text += f" — miglioramento ({abs(delta):.1f}s rispetto al test precedente)"
                            else:
                                obs_text += " — prestazione simile al test precedente"

                        print(f"[TEST] STS: {obs_text}")
                        self._add_observation(obs_text)
                    self._running = False
                    if transition_frames:
                        self._save_frames(transition_frames, "STS", "transizioni")
                    return result, transition_frames, test_id

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

    def _frame_to_base64(self, frame):
        """Converte un frame OpenCV in base64."""
        import base64
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def _save_frames(self, frames_b64: list, test_type: str, label: str):
        """Salva i frame base64 come JPEG nella cartella del diario."""
        import base64, os
        folder = os.path.join(self._output_dir, "test_frames")
        os.makedirs(folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, b64 in enumerate(frames_b64):
            path = os.path.join(folder, f"{ts}_{test_type}_{label}_{i+1}.jpg")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
        print(f"[TEST] Salvati {len(frames_b64)} frame in {folder}/")

    def _analyse_sts_quality(self, transition_frames: list) -> str:
        """Analisi qualitativa VLM sui frame delle transizioni del test STS."""
        prompt = (
            "Stai osservando una persona che esegue il test Sit-to-Stand (5 ripetizioni). "
            "Rispondi in 2-3 frasi: la persona ha usato i braccioli o le mani per alzarsi? "
            "Il movimento è fluido o si notano difficoltà e instabilità?"
        )
        return self._vlm.call_with_images(transition_frames, prompt_text=prompt, max_tokens=100)

    def _analyse_standup(self, standup_frames: list) -> str:
        """Analisi VLM focalizzata sull'alzata dalla sedia nel test TUG."""
        prompt = (
            "Stai osservando una persona che si sta alzando dalla sedia all'inizio del test "
            "Timed Up and Go. Rispondi in 2-3 frasi: la persona ha usato i braccioli o le mani "
            "per spingersi su? Il movimento appare fluido o faticoso?"
        )
        return self._vlm.call_with_images(standup_frames, prompt_text=prompt, max_tokens=100)

    def _analyse_tug_quality(self, tug_frames: list) -> str:
        """Analisi qualitativa VLM sui frame delle fasi del test TUG."""
        prompt = (
            "Stai osservando una persona che esegue il test Timed Up and Go. "
            "Rispondi in 2-3 frasi: l'andatura appare stabile e simmetrica? "
            "La svolta è fluida? Si notano difficoltà o instabilità?"
        )
        return self._vlm.call_with_images(tug_frames, prompt_text=prompt, max_tokens=100)

    def _update_fps(self, detector):
        """Calcola FPS reali e aggiorna il detector."""
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 30:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                real_fps = (len(self._frame_times) - 1) / span
                detector.set_fps(real_fps)