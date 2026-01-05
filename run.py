import cv2
import os
import time
import requests
from dotenv import load_dotenv
from picamera2 import Picamera2
from lcd_driver import LCD

# ================= LINE 設定 =================
load_dotenv()
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("LINE_USER_ID")
PUSH_ENDPOINT = "https://api.line.me/v2/bot/message/push"
# ============================================


def send_text_message(to: str, text: str) -> None:
    """傳送 LINE 文字訊息"""
    if not CHANNEL_ACCESS_TOKEN or not to:
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
    }
    body = {
        "to": to,
        "messages": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }

    try:
        resp = requests.post(PUSH_ENDPOINT, headers=headers,
                             json=body, timeout=10)
        if resp.status_code != 200:
            print("LINE 傳送失敗:", resp.status_code, resp.text)
        else:
            print("LINE 已傳送失敗通知")
    except Exception as e:
        print("LINE 傳送例外:", e)


# ---------------- LCD 文字 ----------------
IDLE_LINE1 = "PLEASE FACE"
IDLE_LINE2 = "THE CAMERA"

VERIFY_LINE1 = "VERIFYING..."

SUCCESS_LINE1 = "ACCESS GRANTED"
SUCCESS_LINE2 = "WELCOME!"

FAIL_LINE1 = "ACCESS DENIED"
FAIL_LINE2 = "TRY AGAIN"
# ----------------------------------------------------------

# ---------------- 參數 ----------------
MODEL_PATH = "me_lbph_model.yml"
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

CONF_THRESHOLD = 70
STABLE_SECONDS = 1.0
RESULT_HOLD_SECONDS = 5.0

LCD_I2C_ADDR = 0x27

EVIDENCE_DIR = "evidence"
# ----------------------------------------------------------

_last_lcd = ("", "")


def lcd_show(lcd, l1, l2):
    global _last_lcd
    l1 = l1[:16].ljust(16)
    l2 = l2[:16].ljust(16)
    if (l1, l2) != _last_lcd:
        lcd.message(l1, 1)
        lcd.message(l2, 2)
        _last_lcd = (l1, l2)


def find_cascade():
    base = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base, CASCADE_FILENAME),
        os.path.join(base, "cascades", CASCADE_FILENAME),
        "/usr/share/opencv4/haarcascades/" + CASCADE_FILENAME,
        "/usr/share/opencv/haarcascades/" + CASCADE_FILENAME,
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find haarcascade_frontalface_default.xml")


def save_fail_evidence(frame_bgr, face_bbox, conf):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    x, y, w, h = face_bbox

    img = frame_bgr.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, f"FAIL conf={conf:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    path = os.path.join(EVIDENCE_DIR, f"fail_{ts}_conf{conf:.1f}.jpg")
    cv2.imwrite(path, img)
    print("[EVIDENCE SAVED]", path)

    # 存證後傳 LINE 文字通知
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        "⚠️ Door access failed!\n"
        f"Time: {ts}"
    )
    send_text_message(USER_ID, msg)


def main():
    lcd = LCD(2, LCD_I2C_ADDR, True)
    lcd_show(lcd, IDLE_LINE1, IDLE_LINE2)

    cascade_path = find_cascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if not os.path.isfile(MODEL_PATH):
        lcd_show(lcd, "MODEL MISSING", "RUN TRAINING")
        raise SystemExit("Model file not found")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    state = "IDLE"
    stable_start = None
    target_result = None
    result_end = None

    print("Running (headless, LCD only)...")

    try:
        while True:
            now = time.time()

            if state == "RESULT_LOCK":
                if now >= result_end:
                    state = "IDLE"
                    stable_start = None
                    target_result = None
                    result_end = None
                    lcd_show(lcd, IDLE_LINE1, IDLE_LINE2)
                time.sleep(0.02)
                continue

            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) == 0:
                state = "IDLE"
                stable_start = None
                target_result = None
                lcd_show(lcd, IDLE_LINE1, IDLE_LINE2)
                time.sleep(0.02)
                continue

            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            label, conf = recognizer.predict(face_img)
            current_result = "OK" if (label == 1 and conf
                                      < CONF_THRESHOLD) else "FAIL"

            if state == "IDLE":
                state = "VERIFYING"
                stable_start = now
                target_result = current_result
                lcd_show(lcd, VERIFY_LINE1, "")

            elif state == "VERIFYING":
                if current_result != target_result:
                    target_result = current_result
                    stable_start = now

                elapsed = now - stable_start

                if elapsed >= STABLE_SECONDS:
                    if target_result == "OK":
                        lcd_show(lcd, SUCCESS_LINE1, SUCCESS_LINE2)
                    else:
                        lcd_show(lcd, FAIL_LINE1, FAIL_LINE2)
                        save_fail_evidence(frame, (x, y, w, h), conf)

                    state = "RESULT_LOCK"
                    result_end = now + RESULT_HOLD_SECONDS

            time.sleep(0.02)

    finally:
        picam2.stop()
        lcd.clear()


if __name__ == "__main__":
    main()
