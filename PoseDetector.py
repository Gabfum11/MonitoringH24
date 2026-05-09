# PoseModule.py (versione corretta — FPS-adaptive)
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
import numpy as np
import math
import json
import statistics
from collections import deque
import time

class PoseDetector:
    def __init__(self, model_path="models/pose_landmarker_full.task"):
        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.landmarker = PoseLandmarker.create_from_options(self.options)
        
        # Risultati
        self.result = None
        self.lmList = []
        
        # Storico per smoothing (dimensioni adattive, ricalcolate in set_fps)
        self.history_size = 10
        self.hip_x_history = deque(maxlen=self.history_size)
        self.hip_y_history = deque(maxlen=self.history_size)
        self.movement_history = deque(maxlen=6)  # ~0.25s: reattiva ma non rumorosa
        self.angle_history = deque(maxlen=5)
        
        # Calibrazione distanza (pixel → metri)
        self.pixels_per_meter = None
        self.distance_calibrating = False
        self._calib_start_hip_x = None
        
        # Metriche qualità
        self.min_confidence_threshold = 0.75
        self.tracking_quality = 0.0
        
        # Frame info
        self.frame_width = 0
        self.frame_height = 0
        self.frame_count = 0
        self.fps = 30
        
        # Detector
        self.last_knee_angle = 0
        self.last_movement = 0
        self.hip_x = 0
        self.hip_y = 0
        self.hip_y_velocity = 0
        self.left_ankle_y = 0
        self.right_ankle_y = 0
        self.left_ankle_x = 0
        self.right_ankle_x = 0
        self.shoulder_mid_x = 0
        self.shoulder_mid_y = 0

        # Storico pixel
        self.hip_x_raw_history = deque(maxlen=self.history_size)
        self.hip_y_raw_history = deque(maxlen=self.history_size)
        self.ankle_y_history_left = deque(maxlen=self.history_size)
        self.ankle_y_history_right = deque(maxlen=self.history_size)
        self.step_activity = 0
        self.posture_history = deque(maxlen=15)
        self.knee_angle_history = deque(maxlen=15)

        # Isteresi walking: contatore di frame consecutivi in WALKING
        self._walking_frames = 0
        self._walking_exit_frames = 0
        # Quanti frame sotto soglia servono per uscire da WALKING (~0.5s)
        self._walking_exit_threshold = 12  # ricalcolato in _update_adaptive_params
        # Cooldown post-alzata: blocca WALKING per ~1s dopo l'ultimo SITTING
        self._sit_cooldown = 0
        self._sit_cooldown_threshold = 24  # ricalcolato in _update_adaptive_params

        # Soglie adattive (ricalcolate in set_fps)
        self._update_adaptive_params()

    def _update_adaptive_params(self):
        """Ricalcola tutte le soglie che dipendono dagli FPS."""
        # Finestra per il calcolo del movement: ~0.4 secondi
        self._movement_window = max(2, int(self.fps * 0.4))
        # Deadzone in pixel: jitter MediaPipe è ~2-5px frame-to-frame,
        # su una finestra di N frame si accumula meno, ma serve comunque una soglia
        self._movement_deadzone = 1
        # Finestra per step_activity: ~0.5 secondi di caviglie
        self._step_window = max(5, int(self.fps * 0.5))
        # Isteresi uscita walking: ~0.5 secondi sotto soglia per tornare a STANDING
        self._walking_exit_threshold = max(6, int(self.fps * 0.5))
        # Cooldown post-alzata: ~1 secondo dopo SITTING prima di permettere WALKING
        self._sit_cooldown_threshold = max(12, int(self.fps * 1.0))
        
        # History size deve essere almeno il doppio della finestra più grande
        min_history = max(self._movement_window, self._step_window) * 2
        if min_history > self.history_size:
            self.set_history_size(min_history)

    # =========================================
    # CALIBRAZIONE DISTANZA (pixel → metri)
    # =========================================
    def start_distance_calibration(self):
        if not self.distance_calibrating:
            self.distance_calibrating = True
            self._calib_start_hip_x = self.hip_x
            self._calib_start_hip_y = self.hip_y
            print("=" * 50)
            print("CALIBRAZIONE DISTANZA AVVIATA")
            print("La persona cammini per 2 metri in linea retta,")
            print("lateralmente rispetto alla telecamera.")
            print("Premi di nuovo [W] quando ha raggiunto i 2 metri.")
            print("=" * 50)
            return False
        else:
            dx = self.hip_x - self._calib_start_hip_x
            dy = self.hip_y - self._calib_start_hip_y
            pixels = math.sqrt(dx**2 + dy**2)
            self.distance_calibrating = False
            
            if pixels < 50:
                print("Troppo poco spostamento, riprova.")
                return False
            
            self.pixels_per_meter = pixels / 2.0
            print("=" * 50)
            print(f"CALIBRAZIONE COMPLETATA")
            print(f"  2 metri = {pixels:.0f} pixel")
            print(f"  1 metro = {self.pixels_per_meter:.1f} pixel")
            print("=" * 50)
            
            self.save_distance_calibration()
            return True

    def save_distance_calibration(self, path="distance_calibration.json"):
        data = {"pixels_per_meter": self.pixels_per_meter}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Calibrazione distanza salvata in {path}")
    
    def load_distance_calibration(self, path="distance_calibration.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.pixels_per_meter = data["pixels_per_meter"]
            print(f"Calibrazione distanza caricata: 1m = {self.pixels_per_meter:.1f} px")
            return True
        except FileNotFoundError:
            print("Nessuna calibrazione distanza trovata. Premi [W] per calibrare.")
            return False
    
    def is_distance_calibrated(self):
        return self.pixels_per_meter is not None
    
    def get_pixels_for_meters(self, meters):
        """Converte metri in pixel. Utile per il TUG test."""
        if self.pixels_per_meter is None:
            return None
        return self.pixels_per_meter * meters
    
    # =========================================
    # VELOCITÀ IN M/S
    # =========================================
    def get_speed_ms(self):
        """Velocità in metri/secondo. Richiede calibrazione distanza.
        
        Usa una finestra temporale invece di frame-to-frame
        per essere stabile a qualsiasi FPS.
        """
        if not self.pixels_per_meter:
            return None
        
        window = max(2, int(self.fps * 0.3))
        if len(self.hip_x_raw_history) < window:
            return None
        
        dx = self.hip_x_raw_history[-1] - self.hip_x_raw_history[-window]
        dy = self.hip_y_raw_history[-1] - self.hip_y_raw_history[-window]
        px_displacement = math.sqrt(dx**2 + dy**2)
        meters = px_displacement / self.pixels_per_meter
        seconds = window / self.fps
        return meters / seconds if seconds > 0 else 0
    
    # =========================================
    # RILEVAMENTO POSE
    # =========================================
    def findPose(self, frame, draw=True, timestamp_ms=None):
        self.frame_height, self.frame_width, _ = frame.shape
        self.frame_count += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        self.result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        self._update_tracking_quality()
        if draw and self.result.pose_landmarks:
            frame = self._draw_landmarks(frame)
        
        return frame
    
    def findPosition(self, frame, draw=False, filtered=True):
        self.lmList = []
        if not self.result.pose_landmarks:
            return self.lmList
        landmarks = self.result.pose_landmarks[0]
        h, w = self.frame_height, self.frame_width
        for id, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            confidence = lm.visibility if hasattr(lm, 'visibility') else 1.0
            self.lmList.append([id, x, y, confidence])
            if draw and confidence > self.min_confidence_threshold:
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                cv2.circle(frame, (x, y), 5, color, cv2.FILLED)
        return self.lmList
    
    # =========================================
    # CALCOLO ANGOLI
    # =========================================
    def findAngle(self, frame, p1, p2, p3, draw=True, filtered=True):
        lmList_to_use = self.lmList
        if len(lmList_to_use) <= max(p1, p2, p3):
            return 0
        
        c1 = lmList_to_use[p1][3]
        c2 = lmList_to_use[p2][3]
        c3 = lmList_to_use[p3][3]
        
        if min(c1, c2, c3) < self.min_confidence_threshold:
            if self.angle_history:
                return self.angle_history[-1]
            return 0
        
        x1, y1 = lmList_to_use[p1][1:3]
        x2, y2 = lmList_to_use[p2][1:3]
        x3, y3 = lmList_to_use[p3][1:3]
        
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle
        
        if filtered:
            self.angle_history.append(angle)
            angle = statistics.mean(self.angle_history)
        
        if draw and frame is not None and len(self.lmList) > max(p1, p2, p3):
            px1, py1 = self.lmList[p1][1:3]
            px2, py2 = self.lmList[p2][1:3]
            px3, py3 = self.lmList[p3][1:3]
            self._draw_angle(frame, px1, py1, px2, py2, px3, py3, angle)
        
        return angle
        
    def _draw_angle(self, frame, x1, y1, x2, y2, x3, y3, angle):
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
        
        for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
            cv2.circle(frame, (x, y), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x, y), 15, (0, 0, 255), 2)
        
        cv2.putText(frame, f"{int(angle)}°", (x2 - 50, y2 + 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    # =========================================
    # RILEVAMENTO MOVIMENTO (FPS-adaptive)
    # =========================================
    def detectWalking(self):
        """
        Rileva il movimento della persona.
        
        Usa una finestra temporale (~0.5s) invece di frame-to-frame
        per essere stabile a qualsiasi FPS. Applica una deadzone
        per ignorare il jitter naturale di MediaPipe.
        """
        if len(self.lmList) < 29:
            return 0
        
        # =========================================
        # COORDINATE PIXEL
        # =========================================
        l_hip_px = self.lmList[23]
        r_hip_px = self.lmList[24]
        hip_x_px = (l_hip_px[1] + r_hip_px[1]) / 2
        hip_y_px = (l_hip_px[2] + r_hip_px[2]) / 2

        # Aggiorna coordinate (sempre pixel)
        self.hip_x = hip_x_px
        self.hip_y = hip_y_px

        # FIX: aggiorna le caviglie PRIMA di appendere alla history
        self.left_ankle_y = self.lmList[27][2]
        self.right_ankle_y = self.lmList[28][2]
        self.left_ankle_x = self.lmList[27][1]
        self.right_ankle_x = self.lmList[28][1]
        self.shoulder_mid_x = (self.lmList[11][1] + self.lmList[12][1]) / 2
        self.shoulder_mid_y = (self.lmList[11][2] + self.lmList[12][2]) / 2

        # Append alle history (ora con valori corretti)
        self.hip_x_raw_history.append(hip_x_px)
        self.hip_y_raw_history.append(hip_y_px)
        self.ankle_y_history_left.append(self.left_ankle_y)
        self.ankle_y_history_right.append(self.right_ankle_y)

        # Velocità verticale anca (per detect sit-to-stand)
        # Finestra di 3 frame: toglie il jitter (±1px → ±8 invece di ±24)
        # ma resta reattiva per rilevare l'alzata (-100 a -400)
        vel_window = 3
        if len(self.hip_y_raw_history) >= vel_window:
            delta_px = self.hip_y_raw_history[-1] - self.hip_y_raw_history[-vel_window]
            dt_sec = vel_window / self.fps
            self.hip_y_velocity = delta_px / dt_sec if dt_sec > 0 else 0
        
        # Step activity: oscillazione caviglie su finestra temporale
        if len(self.ankle_y_history_left) >= self._step_window:
            left_slice = list(self.ankle_y_history_left)[-self._step_window:]
            right_slice = list(self.ankle_y_history_right)[-self._step_window:]
            left_range = max(left_slice) - min(left_slice)
            right_range = max(right_slice) - min(right_slice)
            self.step_activity = max(left_range, right_range)
        else:
            self.step_activity = 0

        # =========================================
        # CALCOLO MOVEMENT (finestra temporale + deadzone)
        # =========================================
        self.hip_x_history.append(hip_x_px)
        self.hip_y_history.append(hip_y_px)
        
        window = self._movement_window
        if len(self.hip_x_history) < window:
            return 0

        # Spostamento su ~0.5 secondi
        dx_px = self.hip_x_history[-1] - self.hip_x_history[-window]
        dy_px = self.hip_y_history[-1] - self.hip_y_history[-window]

        # Deadzone: sotto questa soglia è jitter di MediaPipe
        if abs(dx_px) < self._movement_deadzone:
            dx_px = 0
        if abs(dy_px) < self._movement_deadzone:
            dy_px = 0

        # Normalizza a px/secondo (indipendente da FPS)
        dt_sec = window / self.fps
        if dt_sec > 0:
            dx = dx_px / dt_sec
            dy = dy_px / dt_sec
        else:
            dx = 0
            dy = 0

        # Azzera la y se c'è una transizione sit-to-stand (velocità verticale alta)
        if abs(self.hip_y_velocity) > 90:
            dy = 0

        # Riduci il contributo della y (oscillazione naturale del passo)
        instant_movement = math.sqrt(dx**2 + (dy * 0.75)**2)

        self.movement_history.append(instant_movement)
        return statistics.mean(self.movement_history)

    # =========================================
    # RILEVAMENTO POSTURA (FPS-adaptive)
    # =========================================
    def detect_posture(self, current_test=None):
        lmList_to_use = self.lmList
        if len(lmList_to_use) < 29:
            return 'UNKNOWN'
        
        movement = self.detectWalking()
        self.last_movement = movement

        # Soglie movimento in px/secondo
        if current_test == "TUG":
            movement_threshold = 40
        elif current_test == "STS":
            movement_threshold = 450
        else:
            movement_threshold = 40
        
        # Landmark principali
        nose = lmList_to_use[0]
        l_hip = lmList_to_use[23]
        r_hip = lmList_to_use[24]
        l_knee = lmList_to_use[25]
        r_knee = lmList_to_use[26]
        l_ankle = lmList_to_use[27]
        r_ankle = lmList_to_use[28]
        
        key_points = [nose, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
        avg_confidence = statistics.mean([p[3] for p in key_points])
        
        if avg_confidence < self.min_confidence_threshold or self.tracking_quality < 0.75:
            return 'UNKNOWN'
        
        # Angoli ginocchio
        conf_r = min(self.lmList[24][3], self.lmList[26][3], self.lmList[28][3])
        conf_l = min(self.lmList[23][3], self.lmList[25][3], self.lmList[27][3])

        knee_angle_r = self.findAngle(None, 24, 26, 28, draw=False)
        knee_angle_l = self.findAngle(None, 23, 25, 27, draw=False)

        if conf_r > conf_l:
            knee_angle = knee_angle_r
        elif conf_l > conf_r:
            knee_angle = knee_angle_l
        elif knee_angle_r > 0 and knee_angle_l > 0:
            knee_angle = (knee_angle_r + knee_angle_l) / 2
        else:
            knee_angle = 0

        self.knee_angle_history.append(knee_angle)
        self.last_knee_angle = knee_angle

        # Segnali di cammino:
        # 1. step_activity: oscillazione caviglie (robusto a qualsiasi angolazione)
        # 2. movement: spostamento anca (dipende dalla direzione)
        step_threshold = 15
        stepping = self.step_activity > step_threshold and knee_angle > 140
        moving = movement > movement_threshold

        # Cooldown post-alzata: blocca WALKING per ~1s dopo l'ultimo SITTING
        in_cooldown = self._sit_cooldown < self._sit_cooldown_threshold

        # Il soggetto sta camminando se c'è movement O stepping, e non è in cooldown
        is_walking_now = (moving or stepping) and not in_cooldown

        # Classificazione postura con isteresi
        was_sitting = self.posture_history[-1] == 'SITTING' if self.posture_history else False
        was_walking = self.posture_history[-1] == 'WALKING' if self.posture_history else False

        if was_sitting:
            sit_threshold = 150
        else:
            sit_threshold = 150

        # Prima: controlla SITTING (ha priorità)
        if knee_angle < sit_threshold and (movement < movement_threshold or knee_angle < 120):
            posture = 'SITTING'
            self._walking_frames = 0
            self._walking_exit_frames = 0
            self._sit_cooldown = 0  # resetta il cooldown
        
        # Se ero già in WALKING: isteresi in uscita
        elif was_walking:
            self._sit_cooldown += 1
            if is_walking_now:
                # Continua a camminare, resetta il contatore di uscita
                self._walking_exit_frames = 0
                posture = 'WALKING'
            else:
                # Sotto soglia: conta i frame
                self._walking_exit_frames += 1
                if self._walking_exit_frames >= self._walking_exit_threshold:
                    # Abbastanza tempo sotto soglia → torna a STANDING
                    posture = 'STANDING'
                    self._walking_frames = 0
                    self._walking_exit_frames = 0
                else:
                    # Mantieni WALKING durante la transizione
                    posture = 'WALKING'
        
        # Ingresso in WALKING
        elif is_walking_now:
            self._sit_cooldown += 1
            posture = 'WALKING'
            self._walking_frames += 1
            self._walking_exit_frames = 0
        
        else:
            self._sit_cooldown += 1
            posture = 'STANDING'
            self._walking_frames = 0
            self._walking_exit_frames = 0
                
        self.posture_history.append(posture)
        """
        # DEBUG: stampa 1 volta al secondo
        if self.frame_count % int(self.fps) == 0:
            print(f"[POSTURE DEBUG] mov={movement:.1f} step={self.step_activity:.1f} "
                  f"knee={knee_angle:.0f} cooldown={self._sit_cooldown}/{self._sit_cooldown_threshold} "
                  f"stepping={stepping} moving={moving} exit={self._walking_exit_frames} → {posture}")
        """
        return posture

    # =========================================
    # METRICHE QUALITÀ
    # =========================================
    def _update_tracking_quality(self):
        if not self.result.pose_landmarks:
            self.tracking_quality = 0.0
            return
        
        landmarks = self.result.pose_landmarks[0]
        key_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        confidences = []
        
        for idx in key_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                conf = lm.visibility if hasattr(lm, 'visibility') else 0.5
                confidences.append(conf)
        
        self.tracking_quality = statistics.mean(confidences) if confidences else 0.0
    
    # =========================================
    # DISEGNO
    # =========================================
    def _draw_landmarks(self, frame):
        if not self.result.pose_landmarks:
            return frame
        
        annotated = np.copy(frame)
        h, w = annotated.shape[:2]
        
        for pose_landmarks in self.result.pose_landmarks:
            # Disegna i punti
            for lm in pose_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                conf = lm.visibility if hasattr(lm, 'visibility') else 1.0
                if conf > 0.5:
                    color = (0, 255, 0) if conf > 0.75 else (0, 255, 255)
                    cv2.circle(annotated, (x, y), 4, color, -1)
            
            # Disegna le connessioni
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                (25, 27), (26, 28)
            ]
            for start, end in connections:
                if start < len(pose_landmarks) and end < len(pose_landmarks):
                    x1 = int(pose_landmarks[start].x * w)
                    y1 = int(pose_landmarks[start].y * h)
                    x2 = int(pose_landmarks[end].x * w)
                    y2 = int(pose_landmarks[end].y * h)
                    cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        self._draw_info_overlay(annotated)
        return annotated
    
    def _draw_info_overlay(self, frame):
        y_offset = 30
        quality_color = (0, 255, 0) if self.tracking_quality > 0.7 else \
                        (0, 255, 255) if self.tracking_quality > 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Quality: {self.tracking_quality:.0%}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
    
    # =========================================
    # CONFIGURAZIONE
    # =========================================
    def set_fps(self, fps):
        self.fps = fps
        self._update_adaptive_params()
    
    def set_confidence_threshold(self, threshold):
        self.min_confidence_threshold = threshold
    
    def set_history_size(self, size):
        self.history_size = size
        self.hip_x_history = deque(maxlen=size)
        self.hip_y_history = deque(maxlen=size)
        # movement_history ha la sua dimensione dedicata (6), non va toccata
        self.hip_x_raw_history = deque(maxlen=size)
        self.hip_y_raw_history = deque(maxlen=size)
        self.ankle_y_history_left = deque(maxlen=size)
        self.ankle_y_history_right = deque(maxlen=size)