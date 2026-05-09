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
    frame_times = deque(maxlen=60)

    cv2.namedWindow("TUG Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TUG Debug", area["width"], area["height"])

    print("Premi S per avviare il TUG, Q per uscire")

    while True:
        sct_img = sct.grab(area)
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frame = detector.findPose(frame, draw=True)
        lmList = detector.findPosition(frame, draw=False)
        pose_ok = detector.tracking_quality >= 0.75 and len(lmList) > 0

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
                phase = tug.update(state, detector.hip_x, detector.hip_y,
                                   movement, knee_angle)
                elapsed = time.time() - tug.start_time if tug.start_time else 0

                # Colore per fase
                colors = {
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

                # Log transizioni
                if phase == "FINISHED":
                    result = tug.get_result()
                    print(f"\n{'='*50}")
                    print(f"TUG COMPLETATO")
                    print(f"  Tempo totale: {result['total_time']}s")
                    print(f"  Sit-to-stand: {result['sit_to_stand_time']}s")
                    print(f"  Andata:       {result['walk_forward_time']}s")
                    print(f"  Giravolta:    {result['turn_time']}s")
                    print(f"  Ritorno:      {result['walk_back_time']}s")
                    print(f"  Rischio:      {result['fall_risk_level']}")
                    print(f"{'='*50}\n")
                    active = False
        else:
            cv2.putText(frame, "Persona non rilevata", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # FPS
        now = time.time()
        frame_times.append(now)
        if len(frame_times) >= 2:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            detector.set_fps(fps)
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Istruzioni
        h = frame.shape[0]
        if not active:
            cv2.putText(frame, "Premi S per avviare TUG | Q per uscire",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("TUG Debug", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and pose_ok and not active:
            active = True
            tug.start(detector.hip_x, detector.hip_y)
            print(f"[TUG] Avviato — fase: SIT_TO_STAND")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()