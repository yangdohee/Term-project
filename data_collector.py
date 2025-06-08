import cv2
import numpy as np
from hand_tracker import HandTracker
from utils.landmark_utils import normalize_landmarks

def collect_sequence(label="hello", output_dir="data", frames=300):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        exit()
    tracker = HandTracker()
    sequence = []

    print(f"[INFO] '{label}' 동작을 준비하세요...")

    while cap.isOpened() and len(sequence) < frames:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        lm = tracker.get_hand_landmarks(frame)

        if lm:
            coords = tracker.get_landmark_array(lm, frame.shape)
            norm_coords = normalize_landmarks(coords)
            sequence.append(norm_coords)
            tracker.draw_landmarks(frame, lm)

        cv2.putText(frame, f"Frames Collected: {len(sequence)}/{frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(sequence) == frames:
        np.save(f"{output_dir}/{label}_{np.random.randint(1000)}.npy", np.array(sequence))
        print("[INFO] 저장 완료")

if __name__ == "__main__":
    collect_sequence()