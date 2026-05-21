import time

class SitToStandTest:
    def __init__(self):
        self.test_active = False
        self.reset()
    
    def reset(self):
        self.reps = 0
        self.prev_state = "SITTING"
        self.start_time = None
        self.end_time = None
        self.rep_times = []
        self.current_rep_start = None
        self.knee_angles = []
        self._standing_since = None
        self._min_standing_s = 0.4
        self.transition_just_occurred = False
        self.standup_just_occurred = False
    
    def start(self):
        self.reset()
        self.test_active = True
        self.start_time = time.time()
        self.current_rep_start = time.time()
    
    def update(self, state, knee_angle=0):
        if not self.test_active:
            return self.reps
        
        if knee_angle > 0:
            self.knee_angles.append(knee_angle)

        now = time.time()

        self.transition_just_occurred = False
        self.standup_just_occurred = False

        if state == "STANDING" and self.prev_state != "STANDING":
            self._standing_since = now
            self.standup_just_occurred = True
        elif state == "SITTING" and self.prev_state == "STANDING":
            standing_duration = (now - self._standing_since) if self._standing_since else 0
            if standing_duration >= self._min_standing_s:
                self.reps += 1
                self.transition_just_occurred = True
                if self.current_rep_start:
                    self.rep_times.append(now - self.current_rep_start)
                self.current_rep_start = now
            self._standing_since = None
        elif state != "STANDING":
            self._standing_since = None

        self.prev_state = state
        
        if self.reps >= 5:
            self.end_time = time.time()
            self.test_active = False
        
        return self.reps
    
    def get_result(self):
        """Restituisce i dati per il DB"""
        if not self.end_time:
            return None
        
        total_time = self.end_time - self.start_time
        avg_rep_time = sum(self.rep_times) / len(self.rep_times) if self.rep_times else 0
        avg_knee = sum(self.knee_angles) / len(self.knee_angles) if self.knee_angles else 0
        
        return {
            'total_time': round(total_time, 2),
            'reps_completed': self.reps,
            'avg_rep_time': round(avg_rep_time, 2),
            'avg_knee_angle': round(avg_knee, 2)
        }
    
    def get_time(self):
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return None
