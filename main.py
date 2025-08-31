import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller, Key
import pyautogui
import time
import streamlit as st


st.set_page_config(page_title="AirNav - Gesture Browser Control", layout="wide")


st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">✨ AirNav: Gesture-Based Browser Control ✨</h1>
    <p style="text-align:center; font-size:18px; color:#555;">
        Control your browser using just your hand gestures in real-time! <br>
        <b>Made by Onkar Virakt</b>
    </p>
    """,
    unsafe_allow_html=True
)


left, right = st.columns([1, 2])

with left:
    st.markdown("### 📌 Instructions")
    st.markdown(
        """
        <div style="line-height:1.8; font-size:16px;">
        🔒 Security: System unlocks only when you <b>Smile 😊</b><br><br>
        1️⃣ Press <b>Start</b> to begin gesture recognition.<br>
        2️⃣ Smile at the camera → Unlocks control.<br>
        3️⃣ Once unlocked, move your hand in front of the webcam to control.<br>
        4️⃣ Use gestures:<br>
        &nbsp;&nbsp;&nbsp;👉 Thumb + Middle = Open New Tab <br>
        &nbsp;&nbsp;&nbsp;👉 Fist = Close Tab <br>
        &nbsp;&nbsp;&nbsp;👉 Thumb + Pinky = Scroll Down <br>
        &nbsp;&nbsp;&nbsp;👉 Spread Pinky = Scroll Up <br>
        &nbsp;&nbsp;&nbsp;👉 Thumb + Index = Left Click <br>
        5️⃣ Swipe hand left/right = Switch between tabs.<br><br>
        🛑 Press <b>Stop</b> to end control.
        </div>
        """,
        unsafe_allow_html=True
    )

   
    start_btn = st.button("▶️ Start Control")
    stop_btn = st.button("⏹️ Stop Control")

with right:
    st.markdown("### 🎥 Live Camera Feed")
    stframe = st.empty()


keyboard = Controller()
screen_width, screen_height = pyautogui.size()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = None
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

prev_x, prev_y = 0, 0
smoothing_factor = 0.2
prev_hand_x = None
scrolling = False
scroll_direction = None
click_timer = time.time()
last_action = ""
action_display_time = time.time()
unlocked = False  


def detect_gesture(landmarks):
    thumb, index, middle, ring, pinky = landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]
    middle_thumb_dist = np.hypot(middle.x - thumb.x, middle.y - thumb.y)
    index_thumb_dist = np.hypot(index.x - thumb.x, index.y - thumb.y)
    pinky_thumb_dist = np.hypot(pinky.x - thumb.x, pinky.y - thumb.y)
    pinky_index_dist = np.hypot(pinky.x - index.x, pinky.y - index.y)
    pinky_middle_dist = np.hypot(pinky.x - middle.x, pinky.y - middle.y)
    pinky_ring_dist = np.hypot(pinky.x - ring.x, pinky.y - ring.y)
    fingers_extended = [landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20]]

    if middle_thumb_dist < 0.04:
        return "open_tab"
    elif all(not f for f in fingers_extended):
        return "close_tab"
    elif pinky_thumb_dist < 0.04:
        return "scroll_down"
    elif pinky_index_dist > 0.1 and pinky_middle_dist > 0.1 and pinky_ring_dist > 0.1:
        return "scroll_up"
    elif index_thumb_dist < 0.04:
        return "left_click"
    return None


if start_btn:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_btn:
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #  Unlock check with smile
        if not unlocked:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
                if len(smiles) > 0:  # If smiling detected
                    unlocked = True
                    last_action = "😊 Smile Detected - Control Unlocked"
                    action_display_time = time.time()

            if not unlocked:
                cv2.putText(frame, "🔒 Locked - Please Smile to Unlock", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                continue  # Skip gestures until unlocked

        
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(hand_landmarks.landmark)

            # Cursor Control
            x, y = int(hand_landmarks.landmark[8].x * screen_width), int(hand_landmarks.landmark[8].y * screen_height)
            x = int(prev_x * (1 - smoothing_factor) + x * smoothing_factor)
            y = int(prev_y * (1 - smoothing_factor) + y * smoothing_factor)
            pyautogui.moveTo(x, y, duration=0.08)

            if gesture:
                if gesture == "left_click" and time.time() - click_timer > 0.5:
                    pyautogui.click()
                    click_timer = time.time()
                    last_action = "🖱️ Left Click"
                elif gesture == "open_tab":
                    keyboard.press(Key.ctrl)
                    keyboard.press("t")
                    keyboard.release("t")
                    keyboard.release(Key.ctrl)
                    last_action = "➕ New Tab Opened"
                elif gesture == "close_tab":
                    keyboard.press(Key.ctrl)
                    keyboard.press("w")
                    keyboard.release("w")
                    keyboard.release(Key.ctrl)
                    last_action = "❌ Tab Closed"
                elif gesture == "scroll_down":
                    scrolling = True
                    scroll_direction = "down"
                    last_action = "⬇️ Scrolling Down"
                elif gesture == "scroll_up":
                    scrolling = True
                    scroll_direction = "up"
                    last_action = "⬆️ Scrolling Up"
                action_display_time = time.time()
            else:
                scrolling = False

            if scrolling and (time.time() - action_display_time > 0.05):
                if scroll_direction == "down":
                    pyautogui.scroll(-10)
                elif scroll_direction == "up":
                    pyautogui.scroll(10)
                action_display_time = time.time()

            # Swipe detection for tab switch
            hand_x = hand_landmarks.landmark[0].x
            if prev_hand_x is not None:
                move_threshold = 0.08
                if prev_hand_x - hand_x > move_threshold:
                    keyboard.press(Key.ctrl)
                    keyboard.press(Key.shift)
                    keyboard.press(Key.tab)
                    keyboard.release(Key.tab)
                    keyboard.release(Key.shift)
                    keyboard.release(Key.ctrl)
                    last_action = "⬅️ Switched Left Tab"
                elif hand_x - prev_hand_x > move_threshold:
                    keyboard.press(Key.ctrl)
                    keyboard.press(Key.tab)
                    keyboard.release(Key.tab)
                    keyboard.release(Key.ctrl)
                    last_action = "➡️ Switched Right Tab"

            prev_hand_x = hand_x
            prev_x = x
            prev_y = y

        if last_action and (time.time() - action_display_time < 1.5):
            cv2.putText(frame, last_action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

if stop_btn and cap:
    cap.release()
    st.success("🛑 Gesture Control Stopped")
