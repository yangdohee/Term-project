import customtkinter as ctk
import subprocess
import os
from tkinter import messagebox

# 🔹 venv의 Python 경로 지정
PYTHON_PATH = os.path.join("venv", "Scripts", "python.exe")

# 🔸 현재 실행 중인 main.py 프로세스 추적 변수
main_process = None

# 🔸 수어 수집 + 재학습 실행 함수
def collect_and_train():
    label = entry.get().strip()
    if not label:
        messagebox.showwarning("입력 필요", "수어 이름을 입력해주세요.")
        return
    try:
        subprocess.run([PYTHON_PATH, "batch_collector.py", "--label", label], check=True)
        subprocess.run([PYTHON_PATH, "auto_retrain.py"], check=True)
        messagebox.showinfo("완료", f"'{label}' 수어 수집 및 재학습이 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("오류 발생", f"실행 중 오류가 발생했습니다.\n{e}")

# 🔸 실시간 수어 해석 실행 함수 (중복 실행 방지 포함)
def run_main():
    global main_process
    if main_process is not None and main_process.poll() is None:
        messagebox.showinfo("알림", "캠 실행 중 ..! 대기해주세요.")
        return
    try:
        main_process = subprocess.Popen([PYTHON_PATH, "main.py"])
    except Exception as e:
        messagebox.showerror("실행 오류", str(e))

# 🔸 main.py 프로세스 상태 주기적 확인
def check_main_process():
    global main_process
    if main_process is not None and main_process.poll() is not None:
        main_process = None
    window.after(1000, check_main_process)

# 🔸 customtkinter 설정
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# 🔸 GUI 창 설정
window = ctk.CTk()
window.title("수어 인식 프로젝트 GUI")
window.geometry("460x320")

# 🔸 UI 구성
label = ctk.CTkLabel(master=window, text="새로운 수어 입력", font=("나눔고딕", 14))
label.pack(pady=(20, 5))

entry = ctk.CTkEntry(master=window, width=150, height=30, font=("나눔고딕", 12))
entry.pack(pady=5)

add_button = ctk.CTkButton(master=window, text="➕ 추가하기", command=collect_and_train, corner_radius=20)
add_button.pack(pady=10)

start_button = ctk.CTkButton(master=window, text="▶ 실시간 수어 해석 CAM START", command=run_main, corner_radius=20, fg_color="#38761d", hover_color="#274e13")
start_button.pack(pady=20)

# 🔸 주기적 확인 및 실행
window.after(1000, check_main_process)
window.mainloop()