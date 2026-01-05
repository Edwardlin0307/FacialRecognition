import cv2
import os
import numpy as np

DATA_DIR = "data/me"
MODEL_PATH = "me_lbph_model.yml"

# 讀取所有臉部圖片
images = []
labels = []

for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(DATA_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(1)   # 你的 label，一律設成 1

images = np.array(images)
labels = np.array(labels)

print(f"載入 {len(images)} 張臉部圖片，開始訓練...")

# 建立 LBPH 模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)

recognizer.save(MODEL_PATH)
print(f"訓練完成，模型已存成：{MODEL_PATH}")
