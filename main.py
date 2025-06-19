import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Finger tip IDs (Thumb to Pinky)
tip_ids = [4, 8, 12, 16, 20]

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if lm_list:
                fingers = []

                # Thumb
                if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other fingers
                for id in range(1, 5):
                    if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Middle finger only
                if fingers == [0, 0, 1, 0, 0]:
                    print("Middle finger detected! Shutting down...")
                    os.system('shutdown /s /t 1')  # For Windows
                    # os.system('shutdown now')    # For Linux
                    break

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Middle Finger Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
