# 🔹 main.py (최종)
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import numpy as np
from hand_tracker import HandTracker
from utils.landmark_utils import normalize_landmarks
from model.trainer import LSTMClassifier
import customtkinter as ctk


running = True

def draw_pretty_speech_bubble_cv(image, text, position=(300, 50), font_path="fonts/NanumGothicBold.ttf", font_size=40,
                                  text_color=(0, 0, 0), bg_color=(255, 255, 255, 220), shadow_color=(0, 0, 0, 100)):
    from PIL import ImageFilter

    # OpenCV → PIL
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    font = ImageFont.truetype(font_path, font_size)

    # 말풍선 크기 계산
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding_x, padding_y = 30, 20
    bubble_w = text_width + 2 * padding_x
    bubble_h = text_height + 2 * padding_y

    # 말풍선 + 그림자용 배경
    bubble = Image.new("RGBA", (bubble_w + 8, bubble_h + 8), (0, 0, 0, 0))
    draw_bubble = ImageDraw.Draw(bubble)

    # 그림자 먼저 그림
    shadow_box = (4, 4, bubble_w + 4, bubble_h + 4)
    draw_bubble.rounded_rectangle(shadow_box, radius=20, fill=shadow_color)

    # 그림자를 흐리게
    bubble = bubble.filter(ImageFilter.GaussianBlur(4))

    # 말풍선 본체 그리기 (중앙에 덮어쓰기)
    draw_bubble = ImageDraw.Draw(bubble)
    bubble_box = (0, 0, bubble_w, bubble_h)
    draw_bubble.rounded_rectangle(bubble_box, radius=20, fill=bg_color)

    # 텍스트 삽입
    ascent, descent = font.getmetrics()
    text_x = (bubble_w - text_width) // 2
    text_y = (bubble_h - (ascent + descent)) // 2
    draw_bubble.text((text_x, text_y), text, font=font, fill=text_color)

    # 말풍선을 원본 이미지에 붙임
    image_pil.alpha_composite(bubble, dest=position)

    return cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def update_video():
    global running, label_timer, current_label, sequence
    if not running:
        cap.release()
        cv2.destroyAllWindows()
        window.quit()
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    lm = tracker.get_hand_landmarks(frame)

    if lm:
        coords = tracker.get_landmark_array(lm, frame.shape)
        norm_coords = normalize_landmarks(coords)
        sequence.append(norm_coords)
        tracker.draw_landmarks(frame, lm)

        if len(sequence) == SEQUENCE_LENGTH:
            input_seq = torch.tensor(np.array([sequence]), dtype=torch.float32)
            with torch.no_grad():
                output = model(input_seq)
                pred = torch.argmax(output, dim=1)
                label = label_encoder.inverse_transform(pred.numpy())[0]

            current_label = label
            label_timer = HOLD_TIME
            sequence = []

    if label_timer > 0 and current_label:
        frame = draw_pretty_speech_bubble_cv(frame, current_label, position=(350, 50))
        label_timer -= 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.imgtk = img
    video_label.configure(image=img)

    window.after(10, update_video)

def stop():
    global running
    running = False

model_path = "model/model.pth"
checkpoint = torch.load(model_path, weights_only=False)
model = LSTMClassifier(num_classes=len(checkpoint["label_encoder"].classes_))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
label_encoder = checkpoint["label_encoder"]

SEQUENCE_LENGTH = 30
sequence = []
current_label = ""
label_timer = 0
HOLD_TIME = 5 * 30

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] 카메라를 열 수 없습니다.")
    exit()

tracker = HandTracker()

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")  # 혹은 "green", "dark-blue" 등 테마 선택 가능

window = ctk.CTk()
window.title("실시간 수어 인식")
window.geometry("960x800")
window.configure(fg_color="#f0f0f0")

style = ttk.Style()
style.theme_use("clam")
style.configure("Stop.TButton", font=("\ub9c8\uae08 \uace0\ub515", 11, "bold"), background="#cc0000", foreground="white", padding=5)
style.map("Stop.TButton", background=[("active", "#a40000"), ("pressed", "#800000")])

main_frame = tk.Frame(window, bg="#f0f0f0")
main_frame.pack(fill="both", expand=True)

logo_image = Image.open("communication2.png").resize((60, 60))
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(main_frame, image=logo_photo, bg="#f0f0f0")
logo_label.image = logo_photo
logo_label.pack(pady=(15, 5))

video_frame = tk.Frame(main_frame, bg="#f0f0f0")
video_frame.pack()

video_label = ttk.Label(video_frame)
video_label.grid(row=0, column=0, padx=10, pady=10)

exit_button = ctk.CTkButton(
    video_frame,
    text="종료하기",
    command=stop,
    font=("나눔고딕", 14, "bold"),
    fg_color="#cc0000",
    hover_color="#a40000",
    text_color="white",
    corner_radius=20,
    width=80,
    height=30
)
exit_button.grid(row=1, column=0, sticky="e", padx=20, pady=(5, 20))

update_video()
window.mainloop()