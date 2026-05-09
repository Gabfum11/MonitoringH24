import time
import math
import json
from datetime import date, datetime
from collections import deque


class DailyMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resetta per una nuova giornata"""
        self.date = date.today().isoformat()
        self.start_time = time.time()

        # Contatori frame
        self.frames_total = 0
        self.frames_with_pose = 0
        self.warmup_frames = 20
        self.frame_counter = 0

        # Tempo per stato (in secondi)
        self.time_sitting = 0
        self.time_standing = 0
        self.time_walking = 0

        # Transizioni
        self.num_sit_to_stand = 0
        self.num_stand_to_sit = 0
        self.num_walkings = 0
        self._recent_transitions = deque(maxlen=10)

        # Sit-to-stand timing
        self.sit_to_stand_times = []
        self.sit_to_stand_start = None
        self.slow_transition_threshold = 3.0  # secondi
        self.num_slow_transitions = 0

        # Rilevamento alzata
        self.knee_angle_increasing = []
        self._prev_knee = 0
        self.rising_frames = 0
        # FIX: maxlen limitato a ~3 secondi (ricalcolato quando si conosce il fps)
        self._velocity_maxlen = 90  # ~3s a 30fps, aggiornato da set_fps se serve
        self.recent_velocities = deque(maxlen=self._velocity_maxlen)
        self.rising_in_progress = False
        self.rising_start_time = None  # timestamp di inizio alzata
        self.stable_standing_frames = 0
        
        # Soglie velocità (separate per i due usi):
        #
        # 1. Stabilità post-alzata (usata su media 5 frame nel check rising):
        #    La media 5 frame riduce il jitter quasi a 0, quindi soglia bassa
        self._stable_velocity_threshold = 10
        #
        # 2. Ricerca inizio alzata in _find_rising_start (usata su singoli valori):
        #    I valori dal PoseDetector sono già smoothati su 3 frame (jitter ~±8)
        #    Soglia leggermente più alta per tollerare il jitter residuo
        self._calm_velocity_threshold = 12
        #
        # 3. Calm streak: quanti frame consecutivi "calmi" servono per delimitare
        #    l'inizio dell'alzata (~0.25s, adattivo agli FPS)
        self._calm_streak_required = 6  # ricalcolato se si conosce il fps
        
        # Timeout: se il rising dura più di 5s, qualcosa è andato storto
        self._rising_timeout = 5.0

        # Inattività
        self.longest_inactivity = 0
        self.current_inactivity_start = time.time()

        # Stato precedente
        self.prev_state = "SITTING"
        self._last_known_state = "SITTING"
        self.prev_time = time.time()

        # === EPISODI DI CAMMINO ===
        self.walking_episodes = []
        self.current_walk_start = None
        self.current_walk_distance_px = 0
        self.current_walk_speeds = []
        self.prev_hip_x = None
        self.prev_hip_y = None

        # === EVENTI CON TIMESTAMP ===
        self.events = []

        # === TIME SLOT (ogni 30 minuti) ===
        self.slot_duration = 1800  # 30 minuti in secondi
        self.slot_start = time.time()
        self.slot_sitting = 0
        self.slot_standing = 0
        self.slot_walking = 0
        self.slot_sit_to_stand = 0
        self.completed_slots = []
        # frame
        self._no_tracking_frames = 0

    def _log_event(self, event_type, duration=None, details=None):
        """Registra un evento con timestamp"""
        self.events.append({
            'date': self.date,
            'event_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'event_type': event_type,
            'duration_sec': round(duration, 2) if duration else None,
            'details': json.dumps(details) if details else None
        })

    def _find_rising_start(self, end_time):
        if not self.recent_velocities:
            return None
        
        # Trova il picco di velocità negativa (massima salita)
        peak_idx = 0
        peak_vel = 0
        for i, (t, vel) in enumerate(self.recent_velocities):
            if vel < peak_vel:
                peak_vel = vel
                peak_idx = i
        
        # Dal picco, risali all'indietro fino a trovare calma stabile
        start_idx = peak_idx
        calm_streak = 0
        
        for i in range(peak_idx, -1, -1):
            t, vel = self.recent_velocities[i]
            if abs(vel) < self._calm_velocity_threshold:
                calm_streak += 1
                if calm_streak >= self._calm_streak_required:
                    start_idx = i + calm_streak
                    break
            else:
                calm_streak = 0
                start_idx = i
        
        # Protezione: start_idx non può superare la lunghezza della deque
        if start_idx >= len(self.recent_velocities):
            start_idx = len(self.recent_velocities) - 1
        
        return self.recent_velocities[start_idx][0]

    def _complete_rising(self, current_time):
        """Completa il rilevamento dell'alzata e registra la durata."""
        # Calcola il momento in cui è diventato stabile
        n_stable = min(self.stable_standing_frames, len(self.recent_velocities))
        if n_stable >= 2:
            stable_start_time = self.recent_velocities[-n_stable][0]
        else:
            stable_start_time = current_time

        start_time = self._find_rising_start(current_time)
        if start_time:
            duration = stable_start_time - start_time
            # Sanity check: un'alzata reale non dura più di 10 secondi
            # Se supera, i dati sono corrotti (gap di tracking, velocities vecchie)
            if duration > 10.0:
                print(f"[SIT-TO-STAND] Durata anomala ({duration:.1f}s), scartata")
                self.rising_in_progress = False
                self.rising_start_time = None
                self.stable_standing_frames = 0
                self.recent_velocities.clear()
                return
            if duration > 0.3:  # ignora misurazioni troppo corte
                self.sit_to_stand_times.append(duration)
                if duration > self.slow_transition_threshold:
                    self.num_slow_transitions += 1
                print(f"[SIT-TO-STAND] Durata: {duration:.2f}s")
                self._log_event("SIT_TO_STAND_COMPLETE", duration=duration, details={
                    'duration_sec': round(duration, 2),
                    'slow': duration > self.slow_transition_threshold
                })

        self.rising_in_progress = False
        self.rising_start_time = None
        self.stable_standing_frames = 0
        self.recent_velocities.clear()

    def update(self, state, movement, pose_detected, hip_y_velocity):
        """Chiamata ogni frame"""
        current_time = time.time()
        dt = min(current_time - self.prev_time, 1.0)  # max 1 secondo per frame

        self.frames_total += 1
        if pose_detected:
            self.frames_with_pose += 1
            self._no_tracking_frames = 0

        if not pose_detected:
            self._no_tracking_frames += 1
            
            # Salva lo stato prima che venga sovrascritto con UNKNOWN
            # Solo se era uno stato reale (non UNKNOWN da un gap precedente)
            if self._no_tracking_frames == 1 and self.prev_state != "UNKNOWN":
                self._last_known_state = self.prev_state
            
            # Chiudi episodio di cammino se era aperto
            if self._no_tracking_frames > 5 and self.current_walk_start is not None:
                self.track_walking("STANDING", None, None, None)
            
            # Resetta stato dopo troppi frame senza tracking
            if self._no_tracking_frames > 10:
                self.prev_state = "UNKNOWN"
                # FIX: pulisci il rilevamento alzata — i dati vecchi non sono più validi
                self.recent_velocities.clear()
                self.rising_in_progress = False
                self.stable_standing_frames = 0
            self.prev_time = current_time
            return

        # Accumula tempo per stato (totale giornaliero)
        if state == "SITTING":
            self.time_sitting += dt
        elif state == "STANDING":
            self.time_standing += dt
        elif state == "WALKING":
            self.time_walking += dt

        # Accumula tempo per slot corrente
        if state == "SITTING":
            self.slot_sitting += dt
        elif state == "STANDING":
            self.slot_standing += dt
        elif state == "WALKING":
            self.slot_walking += dt

        # Controlla se lo slot è finito
        if current_time - self.slot_start >= self.slot_duration:
            self.completed_slots.append({
                'date': self.date,
                'slot_start': datetime.fromtimestamp(self.slot_start).isoformat(),
                'slot_end': datetime.now().isoformat(),
                'time_sitting_sec': round(self.slot_sitting, 1),
                'time_standing_sec': round(self.slot_standing, 1),
                'time_walking_sec': round(self.slot_walking, 1),
                'num_sit_to_stand': self.slot_sit_to_stand
            })
            # Reset per il nuovo slot
            self.slot_start = current_time
            self.slot_sitting = 0
            self.slot_standing = 0
            self.slot_walking = 0
            self.slot_sit_to_stand = 0

        # Salta i primi frame (warmup)
        self.frame_counter += 1
        if self.frame_counter <= self.warmup_frames:
            self.prev_state = state
            self.prev_time = current_time
            return

        # Rileva cambi di stato
        if state != self.prev_state:
            self._handle_transition(self.prev_state, state, current_time)

        # === ALZATA ===
        self.recent_velocities.append((current_time, hip_y_velocity))

        if self.prev_state == "SITTING" and state in ("STANDING", "WALKING") and not self.rising_in_progress:
            self.rising_in_progress = True
            self.rising_start_time = current_time
            self.stable_standing_frames = 0

        if self.rising_in_progress:
            # Caso 1: torna a SITTING → alzata fallita
            if state == "SITTING":
                self.rising_in_progress = False
                self.stable_standing_frames = 0
                self.recent_velocities.clear()
            
            # Caso 2: sta camminando → si è chiaramente alzato, completa subito
            elif state == "WALKING":
                self._complete_rising(current_time)
            
            # Caso 3: timeout → troppo tempo, completa con quello che abbiamo
            elif current_time - self.rising_start_time > self._rising_timeout:
                self._complete_rising(current_time)
            
            # Caso 4: in piedi, conta frame stabili con velocità SMOOTHATA
            else:
                # La velocity dal PoseDetector è già su 3 frame (jitter ~±8).
                # La media su 5 frame qui riduce ulteriormente a ~±2.
                # Soglia _stable_velocity_threshold=10: fermo passa, alzata lenta (-15/-20) no.
                n_avg = min(5, len(self.recent_velocities))
                if n_avg >= 3:
                    avg_vel = sum(v for _, v in list(self.recent_velocities)[-n_avg:]) / n_avg
                else:
                    avg_vel = hip_y_velocity

                if abs(avg_vel) < self._stable_velocity_threshold:
                    self.stable_standing_frames += 1
                else:
                    self.stable_standing_frames = 0

                if self.stable_standing_frames >= 8:
                    self._complete_rising(current_time)
                
                # DEBUG alzata
                if self.rising_in_progress:
                    print(f"[RISING] vel={hip_y_velocity:.1f} avg={avg_vel:.1f} stable={self.stable_standing_frames}/8 state={state}")

        # Traccia inattività (soglia adattata alla nuova scala movement)
        if movement > 25:
            inactivity = current_time - self.current_inactivity_start
            if inactivity > self.longest_inactivity:
                self.longest_inactivity = inactivity
            self.current_inactivity_start = current_time

        self.prev_state = state
        self.prev_time = current_time

    def track_walking(self, state, hip_x, hip_y, pixels_per_meter):
        """Tracking episodi di cammino."""
        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time else 0

        if state == "WALKING":
            # Inizio nuovo episodio
            if self.current_walk_start is None:
                print(f"[WALK START] Inizio episodio")
                self.current_walk_start = current_time
                self.current_walk_distance_px = 0
                self.current_walk_speeds = []
                self.prev_hip_x = hip_x
                self.prev_hip_y = hip_y

            # Accumula distanza
            if hip_x is not None and self.prev_hip_x is not None:
                dx = hip_x - self.prev_hip_x
                dy = hip_y - self.prev_hip_y
                step_px = math.sqrt(dx**2 + dy**2)
                if dt > 0 and step_px / dt < 1500:
                    self.current_walk_distance_px += step_px

                    if pixels_per_meter and dt > 0:
                        speed = (step_px / pixels_per_meter) / dt
                        if speed < 5:
                            self.current_walk_speeds.append(speed)

            self.prev_hip_x = hip_x
            self.prev_hip_y = hip_y

        elif self.current_walk_start is not None:
            # Fine episodio: stato non è più WALKING
            duration = current_time - self.current_walk_start
            if duration > 1.0:
                distance_m = (self.current_walk_distance_px / pixels_per_meter) if pixels_per_meter else None
                avg_speed = (sum(self.current_walk_speeds) / len(self.current_walk_speeds)) if self.current_walk_speeds else None

                episode = {
                    'date': self.date,
                    'start_time': datetime.fromtimestamp(self.current_walk_start).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_sec': round(duration, 2),
                    'distance_m': round(distance_m, 2) if distance_m else None,
                    'avg_speed_ms': round(avg_speed, 3) if avg_speed else None,
                }
                self.walking_episodes.append(episode)
                self._log_event("WALK_END", duration=duration, details={
                    'duration_sec': round(duration, 2),
                    'distance_m': round(distance_m, 2) if distance_m else None,
                    'avg_speed_ms': round(avg_speed, 3) if avg_speed else None,
                })
                print(f"[WALK] Durata: {duration:.1f}s"
                      + (f"  Speed: {avg_speed:.2f}m/s" if avg_speed else ""))

            self.current_walk_start = None
            self.prev_hip_x = None
            self.prev_hip_y = None

    def _handle_transition(self, old_state, new_state, current_time):
        """Gestisce le transizioni di stato"""
        print(f"[TRANSIZIONE] {old_state} -> {new_state}")

        if old_state == "SITTING" and new_state != "SITTING":
            self.num_sit_to_stand += 1
            self.slot_sit_to_stand += 1
            self._log_event("SIT_TO_STAND")

        if old_state in ("STANDING", "WALKING") and new_state == "SITTING":
            self.num_stand_to_sit += 1
            self._log_event("STAND_TO_SIT")

        if new_state == "WALKING" and old_state != "WALKING":
            self.num_walkings += 1
            self._log_event("WALK_START")

        self._recent_transitions.append((current_time, f"{old_state}->{new_state}"))

    def get_summary(self):
        """Restituisce il riepilogo da salvare nel DB.
        
        NOTA: i valori temporali sono in SECONDI nonostante le chiavi dicano _min.
        Questo per compatibilità con lo schema DB esistente (colonne _min).
        """
        total_time = time.time() - self.start_time

        avg_sit_to_stand = 0
        max_sit_to_stand = 0
        if self.sit_to_stand_times:
            avg_sit_to_stand = sum(self.sit_to_stand_times) / len(self.sit_to_stand_times)
            max_sit_to_stand = max(self.sit_to_stand_times)

        active_time = self.time_standing + self.time_walking
        total_tracked = self.time_sitting + self.time_standing + self.time_walking
        independence_score = (active_time / total_tracked * 100) if total_tracked > 0 else 0

        final_inactivity = time.time() - self.current_inactivity_start
        if final_inactivity > self.longest_inactivity:
            self.longest_inactivity = final_inactivity

        # Chiudi l'ultimo slot parziale
        if self.slot_sitting + self.slot_standing + self.slot_walking > 0:
            self.completed_slots.append({
                'date': self.date,
                'slot_start': datetime.fromtimestamp(self.slot_start).isoformat(),
                'slot_end': datetime.now().isoformat(),
                'time_sitting_sec': round(self.slot_sitting, 1),
                'time_standing_sec': round(self.slot_standing, 1),
                'time_walking_sec': round(self.slot_walking, 1),
                'num_sit_to_stand': self.slot_sit_to_stand
            })

        return {
            'date': self.date,
            'total_monitoring_min': total_time,
            'time_sitting_min': self.time_sitting,
            'time_standing_min': self.time_standing,
            'time_walking_min': self.time_walking,
            'num_walkings': self.num_walkings,
            'num_sit_to_stand': self.num_sit_to_stand,
            'num_stand_to_sit': self.num_stand_to_sit,
            'avg_sit_to_stand_time': avg_sit_to_stand,
            'max_sit_to_stand_time': max_sit_to_stand,
            'num_slow_transitions': self.num_slow_transitions,
            'longest_inactivity_min': self.longest_inactivity,
            'independence_score': independence_score
        }