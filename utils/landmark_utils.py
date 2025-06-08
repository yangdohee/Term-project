import numpy as np

def normalize_landmarks(landmarks):
    """
    입력: (x, y, z) 좌표 리스트 (21개 관절)
    출력: 기준 좌표로부터의 상대 좌표를 평탄화한 numpy 배열 (1D)
    """
    base_x, base_y, _ = landmarks[0]
    norm_landmarks = [(x - base_x, y - base_y, z) for (x, y, z) in landmarks]
    return np.array(norm_landmarks).flatten()

def landmarks_to_sequence(landmark_seq):
    """
    여러 프레임의 좌표 시퀀스를 정규화하여 하나의 시퀀스로 변환
    """
    return np.stack([normalize_landmarks(lm) for lm in landmark_seq])