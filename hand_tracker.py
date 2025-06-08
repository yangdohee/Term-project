import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.drawing = mp.solutions.drawing_utils

    def get_hand_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    def draw_landmarks(self, image, landmarks):
        self.drawing.draw_landmarks(image, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    def get_landmark_array(self, landmarks, image_shape):
        h, w, _ = image_shape
        return [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks.landmark]