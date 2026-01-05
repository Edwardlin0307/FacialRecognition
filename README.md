# Face Recognition Access Control System
本專題為一套以 Raspberry Pi 為核心的人臉辨識門禁系統，結合 OpenCV 人臉辨識、LCD 顯示器 以及 LINE Bot 即時通知。
系統可在本地端進行人臉辨識，當辨識成功或失敗時即時顯示狀態，並在辨識失敗時透過 LINE 傳送警告訊息給註冊使用者。
📸 使用流程
Step 1｜蒐集人臉資料
python collect.py


系統會自動偵測人臉並拍照

影像會儲存在 data/me/

蒐集完成後自動結束

Step 2｜訓練模型
python train_model.py


使用蒐集到的人臉影像

產生 me_lbph_model.yml

Step 3｜啟動系統
python run.py


系統功能：

偵測人臉後自動進入辨識

LCD 顯示狀態：

PLEASE FACE CAMERA

VERIFYING

ACCESS GRANTED

ACCESS DENIED

辨識失敗時，透過 LINE 傳送警告訊息

按 q 可隨時結束程式

📺 LCD 顯示說明
狀態	LCD 顯示內容
待機	PLEASE FACE CAMERA
辨識中	VERIFYING
成功	ACCESS GRANTED
失敗	ACCESS DENIED
