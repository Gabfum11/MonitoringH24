import time
import math
from datetime import datetime

class TUGTest:
    def __init__(self):
        self.active = False
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.phase = None
        self.prev_hip_x = None
        self.prev_hip_y = None

        # Spostamento cumulativo
        self.cumulative_distance = 0
        self.forward_distance = 0
        self.return_distance = 0 

        # Per rilevare inversione di direzione
        self.timed_deltas = []

        # Fasi per il DB
        self.phase_times = {
            'sit_to_stand': None,
            'walk_forward': None,
            'turn': None,
            'walk_back': None
        }
        self.phase_start_time = None
        self.movement_values = []
        self.knee_angles = []
        self.phase_just_changed = False

    def start(self, hip_x, hip_y):
        """Avvia il test. Non richiede calibrazione."""
        self.reset()
        self.active = True
        self.phase = "SIT_TO_STAND"
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.prev_hip_x = hip_x
        self.prev_hip_y = hip_y

    def update(self, state, hip_x, hip_y, movement=0, knee_angle=0):
        if not self.active:
            return self.phase

        self.phase_just_changed = False

        if movement > 0:
            self.movement_values.append(movement)
        if knee_angle > 0:
            self.knee_angles.append(knee_angle)
        now = time.time()

        if self.prev_hip_x is not None:
            delta_x = hip_x - self.prev_hip_x
            delta_y = hip_y - self.prev_hip_y
            delta = math.sqrt(delta_x**2 + delta_y**2)

            if 0.3 < delta < 50:
                self.cumulative_distance += delta
                self.timed_deltas.append((now, delta_x, delta_y))

                cutoff = now - 3.0
                self.timed_deltas = [(t, dx, dy) for t, dx, dy in self.timed_deltas if t > cutoff]

                if self.phase == "WALK_FORWARD":
                    self.forward_distance += delta
                elif self.phase == "WALK_BACK":
                    self.return_distance += delta

        self.prev_hip_x = hip_x
        self.prev_hip_y = hip_y

        # === TRANSIZIONI ===

        # 1. SIT_TO_STAND → WALK_FORWARD
        if self.phase == "SIT_TO_STAND" and state == "WALKING":
            self.phase_times['sit_to_stand'] = now - self.phase_start_time
            self.phase = "WALK_FORWARD"
            self.phase_start_time = now
            self.forward_distance = 0
            self.timed_deltas = []
            self.phase_just_changed = True
            print(f"[TUG] SIT_TO_STAND → WALK_FORWARD ({self.phase_times['sit_to_stand']:.1f}s)")

        # 2. WALK_FORWARD → TURN
        elif self.phase == "WALK_FORWARD":
            if now - self.phase_start_time > 2.0 and self._detect_direction_change():
                self.phase_times['walk_forward'] = now - self.phase_start_time
                self.phase = "TURN"
                self.phase_start_time = now
                self.timed_deltas = []
                self.phase_just_changed = True
                print(f"[TUG] WALK_FORWARD → TURN ({self.phase_times['walk_forward']:.1f}s)")

        # 3. TURN → WALK_BACK: aspetta almeno 1 secondo nel turn
        elif self.phase == "TURN":
            if now - self.phase_start_time > 1.0 and state == "WALKING":
                self.phase_times['turn'] = now - self.phase_start_time
                self.phase = "WALK_BACK"
                self.phase_start_time = now
                self.return_distance = 0
                self.timed_deltas = []
                self.phase_just_changed = True
                print(f"[TUG] TURN → WALK_BACK ({self.phase_times['turn']:.1f}s)")

        # 4. WALK_BACK → FINISHED
        elif self.phase == "WALK_BACK" and state == "SITTING":
            self.phase_times['walk_back'] = now - self.phase_start_time
            self.phase = "FINISHED"
            self.end_time = now
            self.active = False
            self.phase_just_changed = True
            print(f"[TUG] WALK_BACK → FINISHED ({self.phase_times['walk_back']:.1f}s)")

        # DEBUG: stampa ogni secondo
        if hasattr(self, '_last_debug') and now - self._last_debug > 1.0:
            n_deltas = len(self.timed_deltas)
            elapsed = now - self.phase_start_time
            print(f"[TUG] Phase={self.phase} t={elapsed:.1f}s deltas={n_deltas} "
                f"mov={movement:.1f} state={state} fwd={self.forward_distance:.0f}")
        self._last_debug = now

        return self.phase

    def _detect_direction_change(self):
        """Rileva inversione basata sul tempo, non sui frame."""
        if len(self.timed_deltas) < 5:
            return False

        now = self.timed_deltas[-1][0]

        # Primo secondo vs ultimo secondo
        early = [(dx, dy) for t, dx, dy in self.timed_deltas if t < now - 1.5]
        recent = [(dx, dy) for t, dx, dy in self.timed_deltas if t > now - 1.0]

        if len(early) < 3 or len(recent) < 3:
            return False

        early_avg_x = sum(dx for dx, dy in early) / len(early)
        recent_avg_x = sum(dx for dx, dy in recent) / len(recent)
        early_avg_y = sum(dy for dx, dy in early) / len(early)
        recent_avg_y = sum(dy for dx, dy in recent) / len(recent)

        threshold = 0.2

        if abs(early_avg_x) > threshold and abs(recent_avg_x) > threshold:
            if (early_avg_x > 0) != (recent_avg_x > 0):
                print(f"[TUG] Inversione X! Early={early_avg_x:.2f} Recent={recent_avg_x:.2f}")
                return True

        if abs(early_avg_y) > threshold and abs(recent_avg_y) > threshold:
            if (early_avg_y > 0) != (recent_avg_y > 0):
                print(f"[TUG] Inversione Y! Early={early_avg_y:.2f} Recent={recent_avg_y:.2f}")
                return True

        return False

    def get_result(self):
        if not self.end_time:
            return None

        total_time = self.end_time - self.start_time
        avg_speed = self.cumulative_distance / total_time if total_time > 0 else 0

        return {
            'timestamp': datetime.now().isoformat(),
            'total_time': round(total_time, 2),
            'total_distance_px': round(self.cumulative_distance, 0),
            'avg_speed_px_s': round(avg_speed, 2)
        }