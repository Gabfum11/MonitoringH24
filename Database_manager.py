import sqlite3
from datetime import date, datetime

class DatabaseManager:
    def __init__(self, db_path="monitoring.db"):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        return self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def create_tables(self):
        cursor = self.connect()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_monitoring_min REAL,
                time_sitting_min REAL,
                time_standing_min REAL,
                time_walking_min REAL,
                num_walkings INTEGER,
                num_sit_to_stand INTEGER,
                num_stand_to_sit INTEGER,
                avg_sit_to_stand_time REAL,
                max_sit_to_stand_time REAL,
                num_slow_transitions INTEGER,
                longest_inactivity_min REAL,
                independence_score REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_session (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                test_type TEXT,
                start_time TEXT,
                end_time TEXT,
                completed INTEGER,
                video_source TEXT,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tug_result (
                test_id INTEGER PRIMARY KEY,
                total_time REAL,
                total_distance_px REAL,
                avg_speed_px_s REAL,
                FOREIGN KEY(test_id) REFERENCES test_session(test_id)
            )
        ''')
        # migrazione: aggiunge avg_speed_px_s se la tabella esiste già senza
        try:
            cursor.execute('ALTER TABLE tug_result ADD COLUMN avg_speed_px_s REAL')
        except Exception:
            pass

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sts_result (
                test_id INTEGER PRIMARY KEY,
                total_time REAL,
                reps_completed INTEGER,
                avg_rep_time REAL,
                avg_knee_angle REAL,
                FOREIGN KEY(test_id) REFERENCES test_session(test_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS walking_episode (
                episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_sec REAL,
                distance_m REAL,
                avg_speed_ms REAL,
                cadence REAL,
                symmetry REAL,
                regularity REAL,
                total_steps INTEGER,
                reliability_score INTEGER,
                signal_used TEXT,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_event (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                event_time TEXT,
                event_type TEXT,
                duration_sec REAL,
                details TEXT,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_slot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_date TEXT,
                slot_start TEXT,
                slot_end TEXT,
                time_sitting_sec REAL,
                time_standing_sec REAL,
                time_walking_sec REAL,
                num_sit_to_stand INTEGER,
                FOREIGN KEY(summary_date) REFERENCES daily_summary(date)
            )
        ''')

        self.close()

    # === DAILY SUMMARY ===
    def save_daily_summary(self, summary_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT OR REPLACE INTO daily_summary VALUES (
                :date, :total_monitoring_min,
                :time_sitting_min, :time_standing_min, :time_walking_min,
                :num_walkings, :num_sit_to_stand, :num_stand_to_sit,
                :avg_sit_to_stand_time, :max_sit_to_stand_time,
                :num_slow_transitions,
                :longest_inactivity_min, :independence_score
            )
        ''', summary_data)
        self.close()

    # === TEST SESSION ===
    def create_test_session(self, summary_date, test_type, start_time, video_source=None):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO test_session (summary_date, test_type, start_time, completed, video_source)
            VALUES (?, ?, ?, 0, ?)
        ''', (summary_date, test_type, start_time, video_source))
        test_id = cursor.lastrowid
        self.close()
        return test_id

    def complete_test_session(self, test_id, end_time):
        cursor = self.connect()
        cursor.execute('''
            UPDATE test_session SET end_time = ?, completed = 1 WHERE test_id = ?
        ''', (end_time, test_id))
        self.close()

    # === TUG RESULT ===
    def save_tug_result(self, test_id, tug_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO tug_result (test_id, total_time, total_distance_px, avg_speed_px_s)
            VALUES (:test_id, :total_time, :total_distance_px, :avg_speed_px_s)
        ''', {**tug_data, 'test_id': test_id})
        self.close()

    # === QUERY TEST RESULTS ===
    def get_tug_results(self, start_date, end_date):
        """Restituisce i risultati TUG completati in un range di date (YYYY-MM-DD)."""
        cursor = self.connect()
        cursor.execute('''
            SELECT ts.summary_date, tr.total_time, tr.total_distance_px, tr.avg_speed_px_s
            FROM test_session ts
            JOIN tug_result tr ON ts.test_id = tr.test_id
            WHERE ts.summary_date BETWEEN ? AND ? AND ts.completed = 1
            ORDER BY ts.start_time
        ''', (start_date, end_date))
        rows = cursor.fetchall() 
        self.close()
        return [{'date': r[0], 'total_time': r[1],
                 'total_distance_px': r[2], 'avg_speed_px_s': r[3]} for r in rows]

    def get_sts_results(self, start_date, end_date):
        """Restituisce i risultati STS completati in un range di date (YYYY-MM-DD)."""
        cursor = self.connect()
        cursor.execute('''
            SELECT ts.summary_date, sr.total_time, sr.reps_completed, sr.avg_rep_time, sr.avg_knee_angle
            FROM test_session ts
            JOIN sts_result sr ON ts.test_id = sr.test_id
            WHERE ts.summary_date BETWEEN ? AND ? AND ts.completed = 1
            ORDER BY ts.start_time
        ''', (start_date, end_date))
        rows = cursor.fetchall()
        self.close()
        return [{'date': r[0], 'total_time': r[1], 'reps_completed': r[2],
                 'avg_rep_time': r[3], 'avg_knee_angle': r[4]} for r in rows]

    # === STS RESULT ===
    def save_sts_result(self, test_id, sts_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO sts_result VALUES (
                :test_id, :total_time, :reps_completed, :avg_rep_time, :avg_knee_angle
            )
        ''', {**sts_data, 'test_id': test_id})
        self.close()

    # === WALKING EPISODE ===
    def save_walking_episode(self, episode_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO walking_episode
            (summary_date, start_time, end_time, duration_sec, distance_m, avg_speed_ms,
             cadence, symmetry, regularity, total_steps)
            VALUES (:date, :start_time, :end_time, :duration_sec, :distance_m, :avg_speed_ms,
                    :cadence, :symmetry, :regularity, :total_steps)
        ''', episode_data)
        self.close()

    # === ACTIVITY EVENT ===
    def save_activity_event(self, event_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO activity_event
            (summary_date, event_time, event_type, duration_sec, details)
            VALUES (:date, :event_time, :event_type, :duration_sec, :details)
        ''', event_data)
        self.close()

    # === TIME SLOT ===
    def save_time_slot(self, slot_data):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO time_slot
            (summary_date, slot_start, slot_end, time_sitting_sec, time_standing_sec,
             time_walking_sec, num_sit_to_stand)
            VALUES (:date, :slot_start, :slot_end, :time_sitting_sec, :time_standing_sec,
                    :time_walking_sec, :num_sit_to_stand)
        ''', slot_data)
        self.close()
