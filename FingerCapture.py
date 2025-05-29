import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time 
from DiffusionDet.videos_deno import *

# MediaPipe 手部偵測設定
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # 假設只追蹤一隻手
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 定義五個指尖的索引
FINGER_TIPS = {
    "thumb": mp_hands.HandLandmark.THUMB_TIP,
    "index": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_TIP,
    "pinky": mp_hands.HandLandmark.PINKY_TIP,
}

all_finger_coordinates_history = []


# ✅ 自動偵測可用攝影機
def find_working_camera_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return -1

camera_index = find_working_camera_index()
if camera_index == -1:
    print("❌ 找不到可用的攝影機，請確認是否已連接或啟用。")
    exit()
else:
    print(f"✅ 使用攝影機索引：{camera_index}")

cap = cv2.VideoCapture(camera_index)

print("按下 'q' 鍵結束錄製並匯出 Excel。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像。")
        break

    # 翻轉影像，使其像是照鏡子 (可選)
    frame = cv2.flip(frame, 1)

    # 轉換 BGR 到 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 處理影像，偵測手部關鍵點
    results = hands.process(rgb_frame)

    # 儲存當前幀的指尖座標 (字典形式)
    current_frame_finger_data = {}
    current_frame_finger_data['timestamp'] = time.time() # 可以加入時間戳

    # 在影像上繪製手部關鍵點並提取指尖座標
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 繪製手部骨架
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, c = frame.shape # 獲取影像尺寸

            # 遍歷每個指尖，提取並儲存座標
            for finger_name, landmark_id in FINGER_TIPS.items():
                tip_landmark = hand_landmarks.landmark[landmark_id]
                cx, cy = int(tip_landmark.x * w), int(tip_landmark.y * h)

                # 將座標儲存到當前幀的字典中
                current_frame_finger_data[f'{finger_name}_x'] = cx
                current_frame_finger_data[f'{finger_name}_y'] = cy

                # 在指尖畫一個圓點並顯示名稱
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1) # 綠色實心圓
                cv2.putText(frame, finger_name, (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 如果有手被偵測到，才將資料加入歷史記錄
    if results.multi_hand_landmarks:
        all_finger_coordinates_history.append(current_frame_finger_data)

    # 顯示處理後的影像
    cv2.imshow('Multi-Finger Tracking (Press Q to Quit)', frame)

    # 按下 'q' 鍵結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()

print("\n--- 所有指尖路徑座標 (字典資料) ---")
if all_finger_coordinates_history:
    # 這裡只印出前幾筆和後幾筆，避免資料過多
    num_to_print = min(5, len(all_finger_coordinates_history))
    for i in range(num_to_print):
        print(all_finger_coordinates_history[i])
    if len(all_finger_coordinates_history) > 2 * num_to_print:
        print("...")
        for i in range(len(all_finger_coordinates_history) - num_to_print, len(all_finger_coordinates_history)):
            print(all_finger_coordinates_history[i])
else:
    print("沒有擷取到指尖座標資料。")
print(f"總共收集到 {len(all_finger_coordinates_history)} 幀的指尖座標資料。")

if all_finger_coordinates_history:
    df = pd.DataFrame(all_finger_coordinates_history)
    output_filename = 'all_finger_tips_path.xlsx'
    df.to_excel(output_filename, index=False)
    print(f"所有指尖路徑座標已匯出到 {output_filename}")
else:
    print("沒有擷取到指尖座標資料")