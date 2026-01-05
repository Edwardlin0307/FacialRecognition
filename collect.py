import cv2
import os
import time
from picamera2 import Picamera2

# --------------- 設定區 ---------------
DATA_DIR = "data/me"      # 儲存臉部影像的資料夾
MAX_IMAGES = 100          # 自動蒐集張數
CAPTURE_INTERVAL = 0.5   # 拍照間隔（秒）
CASCADE_PATH = "haarcascade_frontalface_default.xml"
# --------------------------------------

# 建立資料夾
os.makedirs(DATA_DIR, exist_ok=True)

# 載入 Haar Cascade（本地）
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("❌ 無法載入 Haar Cascade")
    raise SystemExit

# 初始化 Picamera2
picam2 = Picamera2()

# 設定相機輸出格式（OpenCV 吃 RGB）
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

count = 0
last_capture_time = 0

print("📸 使用 Picamera2 開始自動蒐集臉部資料（無畫面）")

while True:
    # 取得影像（numpy array）
    frame = picam2.capture_array()

    # 轉灰階（Haar Cascade 必須）
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 偵測人臉
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        if count < MAX_IMAGES and (current_time -
                                   last_capture_time) > CAPTURE_INTERVAL:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            file_path = os.path.join(DATA_DIR, f"me_{count:03d}.png")
            cv2.imwrite(file_path, face_img)

            print(f"✅ Auto saved: {file_path}")
            count += 1
            last_capture_time = current_time

        break  # 同一幀只存一張臉

    if count >= MAX_IMAGES:
        print("🎉 臉部資料蒐集完成")
        break

# 停止相機
picam2.stop()
