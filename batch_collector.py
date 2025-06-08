import os
import time
import cv2
import numpy as np
import argparse
from hand_tracker import HandTracker
from utils.landmark_utils import normalize_landmarks

FRAMES_PER_SEQUENCE = 30  # 시퀀스당 프레임 수
SEQUENCES_PER_LABEL = 10  # 수어당 시퀀스 개수
WAIT_SECONDS = 10         # 대기 시간

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 🔸 argparse로 외부에서 수어 라벨을 받아 실행 가능하게
parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="추가할 수어 라벨명")
args = parser.parse_args()

LABEL = args.label

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 카메라를 열 수 없습니다.")
    exit()

tracker = HandTracker()
print(f"[INFO] '{LABEL}' 수어 수집 시작 - 총 {SEQUENCES_PER_LABEL}개")

for i in range(SEQUENCES_PER_LABEL):
    print(f"[INFO] {LABEL} 동작 {i+1}/{SEQUENCES_PER_LABEL} 준비하세요... ({WAIT_SECONDS}초 대기)")
    time.sleep(WAIT_SECONDS)

    sequence = []
    collected = 0

    while collected < FRAMES_PER_SEQUENCE:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        lm = tracker.get_hand_landmarks(frame)

        if lm:
            coords = tracker.get_landmark_array(lm, frame.shape)
            norm_coords = normalize_landmarks(coords)
            sequence.append(norm_coords)
            collected += 1
            tracker.draw_landmarks(frame, lm)

        cv2.putText(frame, f"Seq [{i+1}/{SEQUENCES_PER_LABEL}] Frame {collected}/{FRAMES_PER_SEQUENCE}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    sequence = np.array(sequence)
    filename = f"{LABEL}_{int(time.time())}.npy"
    np.save(os.path.join(output_dir, filename), sequence)
    print(f"[INFO] 저장됨: {filename}")

print("[INFO] 수어 '%s' 수집 완료." % LABEL)
cap.release()
cv2.destroyAllWindows()