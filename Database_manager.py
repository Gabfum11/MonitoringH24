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
            CREATE TABLE IF NOT EXISTS tug_result (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_time REAL,
                total_distance_torsos REAL DEFAULT 0,
                tracking_lost_seconds REAL DEFAULT 0,
                vlm_analysis_time REAL,
                vlm_analysis TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sts_result (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                completed INTEGER DEFAULT 0,
                total_time REAL,
                reps_completed INTEGER,
                vlm_analysis_time REAL,
                vlm_analysis TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sts_rep (
                test_id INTEGER,
                numero_rep INTEGER,
                tempo_rep REAL,
                PRIMARY KEY(test_id, numero_rep),
                FOREIGN KEY(test_id) REFERENCES sts_result(test_id)
            )
        ''')

        self._migrate()
        self.close()

    def _migrate(self):
        """Aggiunge colonne mancanti per retrocompatibilità con DB esistenti."""
        cursor = self.conn.cursor()
        existing_tug = {row[1] for row in cursor.execute("PRAGMA table_info(tug_result)")}

        # Se il DB esistente ha le colonne legacy in pixel, ricostruisco la tabella
        # convertendo i valori storici in lunghezze di busto (dove possibile)
        if 'total_distance_px' in existing_tug:
            has_torso = 'baseline_torso_length_px' in existing_tug
            cursor.execute('''
                CREATE TABLE tug_result_new (
                    test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    total_time REAL,
                    total_distance_torsos REAL DEFAULT 0,
                    tracking_lost_seconds REAL DEFAULT 0,
                    vlm_analysis TEXT
                )
            ''')
            torso_expr = ('CASE WHEN baseline_torso_length_px > 0 '
                          'THEN total_distance_px / baseline_torso_length_px ELSE 0 END') \
                         if has_torso else '0'
            cursor.execute(f'''
                INSERT INTO tug_result_new (test_id, timestamp, total_time,
                                            total_distance_torsos, tracking_lost_seconds, vlm_analysis)
                SELECT test_id, timestamp, total_time,
                       {torso_expr},
                       COALESCE(tracking_lost_seconds, 0),
                       vlm_analysis
                FROM tug_result
            ''')
            cursor.execute("DROP TABLE tug_result")
            cursor.execute("ALTER TABLE tug_result_new RENAME TO tug_result")
            existing_tug = {row[1] for row in cursor.execute("PRAGMA table_info(tug_result)")}

        for col, typ in [('vlm_analysis', 'TEXT'), ('timestamp', 'TEXT'),
                         ('tracking_lost_seconds', 'REAL DEFAULT 0'),
                         ('total_distance_torsos', 'REAL DEFAULT 0'),
                         ('vlm_analysis_time', 'REAL')]:
            if col not in existing_tug:
                cursor.execute(f"ALTER TABLE tug_result ADD COLUMN {col} {typ}")

        existing_sts = {row[1] for row in cursor.execute("PRAGMA table_info(sts_result)")}
        for col, typ in [('vlm_analysis', 'TEXT'), ('timestamp', 'TEXT'), ('completed', 'INTEGER'),
                         ('vlm_analysis_time', 'REAL')]:
            if col not in existing_sts:
                cursor.execute(f"ALTER TABLE sts_result ADD COLUMN {col} {typ}")

    # === TUG RESULT ===
    def save_tug_result(self, tug_data, timestamp):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO tug_result (timestamp, total_time)
            VALUES (?, ?)
        ''', (
            timestamp,
            tug_data['total_time'],
        ))
        test_id = cursor.lastrowid
        self.close()
        return test_id

    def update_tug_vlm_analysis(self, test_id, vlm_analysis, vlm_analysis_time=None):
        cursor = self.connect()
        cursor.execute(
            "UPDATE tug_result SET vlm_analysis = ?, vlm_analysis_time = ? WHERE test_id = ?",
            (vlm_analysis, vlm_analysis_time, test_id)
        )
        self.close()

    def get_tug_results(self, start_date, end_date):
        """Restituisce i risultati TUG in un range di date (YYYY-MM-DD)."""
        cursor = self.connect()
        cursor.execute('''
            SELECT timestamp, total_time
            FROM tug_result
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_date, end_date + 'T99'))
        rows = cursor.fetchall()
        self.close()
        return [{'date': r[0][:10], 'total_time': r[1]} for r in rows]

    # === STS RESULT ===
    def save_sts_result(self, sts_data, timestamp, completed=1):
        cursor = self.connect()
        cursor.execute('''
            INSERT INTO sts_result (timestamp, completed, total_time, reps_completed)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, completed, sts_data.get('total_time'), sts_data.get('reps_completed')))
        test_id = cursor.lastrowid
        self.close()
        return test_id

    def save_sts_reps(self, test_id, rep_times):
        cursor = self.connect()
        for i, t in enumerate(rep_times, start=1):
            cursor.execute(
                "INSERT INTO sts_rep (test_id, numero_rep, tempo_rep) VALUES (?, ?, ?)",
                (test_id, i, round(t, 3))
            )
        self.close()

    def get_sts_reps(self, test_id):
        cursor = self.connect()
        cursor.execute(
            "SELECT numero_rep, tempo_rep FROM sts_rep WHERE test_id = ? ORDER BY numero_rep",
            (test_id,)
        )
        rows = cursor.fetchall()
        self.close()
        return [{'numero_rep': r[0], 'tempo_rep': r[1]} for r in rows]

    def update_sts_vlm_analysis(self, test_id, vlm_analysis, vlm_analysis_time=None):
        cursor = self.connect()
        cursor.execute(
            "UPDATE sts_result SET vlm_analysis = ?, vlm_analysis_time = ? WHERE test_id = ?",
            (vlm_analysis, vlm_analysis_time, test_id)
        )
        self.close()

    def get_sts_results(self, start_date, end_date):
        """Restituisce i risultati STS in un range di date (YYYY-MM-DD), inclusi i tempi delle singole ripetizioni."""
        cursor = self.connect()
        cursor.execute('''
            SELECT test_id, timestamp, total_time, reps_completed
            FROM sts_result
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_date, end_date + 'T99'))
        rows = cursor.fetchall()
        results = []
        for test_id, timestamp, total_time, reps_completed in rows:
            cursor.execute('''
                SELECT numero_rep, tempo_rep
                FROM sts_rep
                WHERE test_id = ?
                ORDER BY numero_rep
            ''', (test_id,))
            rep_rows = cursor.fetchall()
            rep_times = [{'numero': n, 'tempo': round(t, 2)} for n, t in rep_rows]
            results.append({
                'date': timestamp[:10],
                'total_time': total_time,
                'reps_completed': reps_completed,
                'rep_times': rep_times,
            })
        self.close()
        return results
