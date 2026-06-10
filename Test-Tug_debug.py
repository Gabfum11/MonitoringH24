# test_tug_debug.py
"""
Debug visivo del TUG test. Mostra la finestra con:
- Pose detection
- Fase TUG corrente
- Movement, knee angle, distanza
- Transizioni in tempo reale

Uso:
    python test_tug_debug.py
    Premi S per avviare il TUG, Q per uscire.
"""

import cv2
import os
import mss
import numpy as np
import time
from collections import deque

import PoseDetector as pm
from TugTest import TUGTest


def main():
    area = {"top": 270, "left": 10, "width": 900, "height": 520}
    sct = mss.mss()
    detector = pm.PoseDetector()
    detector.set_fps(20)
    detector.set_confidence_threshold(0.6)
    tug = TUGTest()

    active = False
    waiting_for_seat = False
    seated_since = None
    SEAT_HOLD = 1.0  # secondi di SITTING stabile prima di avviare
    frame_times = deque(maxlen=60)

    cv2.namedWindow("TUG Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TUG Debug", area["width"], area["height"])

    # Video writer e cartella frame: inizializzati alla pressione di S
    video_writer = None
    video_path = None
    frames_folder = None
    prev_phase = None
    standup_captured = False

    print("Premi S per avviare il TUG, Q per uscire")

    while True:
        sct_img = sct.grab(area)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame = detector.findPose(frame, draw=True)
        lmList = detector.findPosition(frame, draw=False)
        pose_ok = detector.tracking_quality >= 0.75 and len(lmList) > 0

        # Info tracking (sempre visibili)
        pose_detected = len(lmList) > 0
        left_hip_conf = lmList[23][3] if len(lmList) > 23 else 0
        right_hip_conf = lmList[24][3] if len(lmList) > 24 else 0
        hips_visible = left_hip_conf > 0.5 and right_hip_conf > 0.5

        tq_color = (0, 255, 0) if pose_ok else (0, 200, 255) if hips_visible else (0, 0, 255)
        cv2.putText(frame, f"Quality: {detector.tracking_quality:.2f}  pose_ok: {pose_ok}",
                    (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tq_color, 2)
        cv2.putText(frame, f"Hip L: {left_hip_conf:.2f}  Hip R: {right_hip_conf:.2f}  hips_visible: {hips_visible}",
                    (10, frame.shape[0] - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tq_color, 2)
        cv2.putText(frame, f"Pose detected: {pose_detected}",
                    (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tq_color, 2)

        if pose_ok:
            state = detector.detect_posture("TUG" if active else None)
            knee_angle = detector.last_knee_angle
            movement = detector.last_movement

            # Disegna angoli
            detector.findAngle(frame, 24, 26, 28, draw=True, filtered=False)
            detector.findAngle(frame, 23, 25, 27, draw=True, filtered=False)

            # Info base
            cv2.putText(frame, f"Stato: {state}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Movement: {movement:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Knee: {knee_angle:.0f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Step: {detector.step_activity:.1f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if active:
                # Cattura il frame appena prima dell'avvio del timer (transizione READY → SIT_TO_STAND)
                if (tug.phase == "READY" and state != "SITTING"
                        and not standup_captured and frames_folder is not None):
                    cv2.imwrite(os.path.join(frames_folder, "alzata_1.jpg"),
                                frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    standup_captured = True
                    print(f"[TUG-DEBUG] Catturato frame alzata")

                phase = tug.update(state, detector.hip_x, detector.hip_y,
                                   movement, knee_angle, detector.torso_length)
                elapsed = time.time() - tug.start_time if tug.start_time else 0

                # Cattura frame ad ogni transizione di fase
                if (tug.phase_just_changed and prev_phase != tug.phase
                        and frames_folder is not None):
                    fname = f"fase_{tug.phase}.jpg"
                    cv2.imwrite(os.path.join(frames_folder, fname),
                                frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print(f"[TUG-DEBUG] Catturato frame per fase {tug.phase}")
                prev_phase = tug.phase

                # Colore per fase
                colors = {
                    "READY": (200, 200, 200),
                    "SIT_TO_STAND": (0, 165, 255),
                    "WALK_FORWARD": (0, 255, 0),
                    "TURN": (0, 255, 255),
                    "WALK_BACK": (255, 0, 0),
                    "FINISHED": (0, 0, 255)
                }
                color = colors.get(phase, (255, 255, 255))

                cv2.putText(frame, f"TUG: {phase}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Tempo: {elapsed:.1f}s", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Dist fwd: {tug.forward_distance:.0f}px", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, f"Dist back: {tug.return_distance:.0f}px", (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, f"Hip: ({detector.hip_x:.0f}, {detector.hip_y:.0f})",
                            (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                # Info per la calibrazione della soglia
                if tug.baseline_hip_y is not None:
                    delta = tug.baseline_hip_y - detector.hip_y
                    cv2.putText(frame, f"Busto: {tug.baseline_torso_length:.0f}px  soglia: {tug.hip_lift_threshold_px:.1f}px  Δ: {delta:+.1f}",
                                (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                else:
                    cv2.putText(frame, f"Baseline in costruzione... (torso ora: {detector.torso_length:.0f}px)",
                                (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                # Log transizioni
                if phase == "FINISHED":
                    result = tug.get_result()
                    pt = tug.phase_times
                    print(f"\n{'='*50}")
                    print(f"TUG COMPLETATO")
                    print(f"  Tempo totale: {result['total_time']}s")
                    print(f"  Sit-to-stand: {pt.get('sit_to_stand', 0):.2f}s")
                    print(f"  Andata:       {pt.get('walk_forward', 0):.2f}s")
                    print(f"  Giravolta:    {pt.get('turn', 0):.2f}s")
                    print(f"  Ritorno:      {pt.get('walk_back', 0):.2f}s")
                    print(f"{'='*50}\n")
                    active = False
        else:
            if not pose_detected:
                cv2.putText(frame, "Persona non rilevata", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tracking parziale (anche OK)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # FPS
        now = time.time()
        frame_times.append(now)
        if len(frame_times) >= 2:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            detector.set_fps(fps)
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Avvio differito: aspetta SITTING stabile per SEAT_HOLD secondi
        if waiting_for_seat and pose_ok:
            if state == "SITTING" and detector.last_knee_angle > 0:
                if seated_since is None:
                    seated_since = time.time()
                elif time.time() - seated_since >= SEAT_HOLD:
                    waiting_for_seat = False
                    seated_since = None
                    active = True
                    tug.start(detector.hip_x, detector.hip_y)
                    print(f"[TUG] Persona seduta stabile — fase: READY (baseline in costruzione)")
                    print(f"[DEBUG] t,phase,hip_y,torso,baseline,threshold,delta,state,knee,tracking_info")
            else:
                seated_since = None

        # Stampa dati per ogni frame durante TUG attivo
        if active:
            t_now = time.time()
            t_rel = t_now - (tug.start_time if tug.start_time else t_now)
            baseline_str = f"{tug.baseline_hip_y:.1f}" if tug.baseline_hip_y is not None else "-"
            thr_str = f"{tug.hip_lift_threshold_px:.1f}" if tug.hip_lift_threshold_px is not None else "-"
            delta_str = f"{tug.baseline_hip_y - detector.hip_y:+.1f}" if tug.baseline_hip_y is not None and pose_detected else "-"
            knee_str = f"{detector.last_knee_angle:.0f}" if pose_detected else "-"
            state_str = state if pose_ok else "no_pose"
            hip_y_str = f"{detector.hip_y:.1f}" if pose_detected else "-"
            torso_str = f"{detector.torso_length:.1f}" if pose_detected else "-"
            print(f"[DEBUG] {t_rel:+.2f},{tug.phase},{hip_y_str},"
                  f"{torso_str},{baseline_str},{thr_str},{delta_str},{state_str},{knee_str},"
                  f"quality={detector.tracking_quality:.2f},hip_L={left_hip_conf:.2f},hip_R={right_hip_conf:.2f},pose_ok={pose_ok},hips_visible={hips_visible}")

        # Istruzioni
        h = frame.shape[0]
        if not active and not waiting_for_seat:
            cv2.putText(frame, "Premi S per avviare TUG | Q per uscire",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        elif waiting_for_seat:
            cv2.putText(frame, "In attesa che la persona si sieda...",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("TUG Debug", frame)

        if video_writer is not None:
            video_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not active and not waiting_for_seat:
            waiting_for_seat = True
            standup_captured = False
            prev_phase = None
            h, w = frame.shape[:2]
            ts = time.strftime('%Y%m%d_%H%M%S')
            video_path = f"tug_debug_{ts}.mp4"
            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                20.0,
                (w, h),
            )
            frames_folder = os.path.join("debug_frames", f"{ts}_TUG")
            os.makedirs(frames_folder, exist_ok=True)
            print(f"[TUG] Comando ricevuto. Vai a sederti sulla sedia del test...")
            print(f"[TUG] Video in registrazione: {video_path}")
            print(f"[TUG] Frame salvati in: {frames_folder}/")
        elif key == ord('q'):
            break

    if video_writer is not None:
        video_writer.release()
        print(f"[TUG] Video salvato: {video_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()