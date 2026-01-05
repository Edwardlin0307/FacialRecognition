# Face Recognition Access Control System
本專題為一套以 Raspberry Pi 為核心的人臉辨識門禁系統，結合 OpenCV 人臉辨識、LCD 顯示器 以及 LINE Bot 即時通知。
系統可在本地端進行人臉辨識，當辨識成功或失敗時即時顯示狀態，並在辨識失敗時透過 LINE 傳送警告訊息給註冊使用者。


使用流程

Step 1｜將全部檔案下載進同一個資料夾，並且填上.env檔案裡的 LINE_CHANNEL_ACCESS_TOKEN和LINE_USER_ID 


Step 2｜蒐集人臉資料
執行collect.py

將面部正對鏡頭，系統會自動偵測人臉並拍照，當拍攝達一定數量後系統便會自行停止，影像會儲存在 data/me/

Step 3｜訓練模型
執行train_model.py

會將蒐集到的人臉影像產生 me_lbph_model.yml

Step 4｜啟動系統
執行run.py

將鏡頭正對人臉，偵測到人臉後自動進入辨識(必須非常正對否則無法進行辨識)

待機:	      PLEASE FACE CAMERA

辨識中:      VERIFYING

成功:	      ACCESS GRANTED

失敗:	      ACCESS DENIED

辨識失敗時，透過 LINE 傳送警告訊息，並且將辨識失敗的面部進行拍照並且存到資料夾evidence

按 q 可隨時結束程式

