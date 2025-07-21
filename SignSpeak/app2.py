import hand_detector2 as hdm
import cv2
import numpy as np
import joblib
import asyncio
import websockets
import base64
from gtts import gTTS
import io
import pygame

pygame.mixer.init()

# Load trained model
model = joblib.load("hand_sign_model.pkl")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = hdm.handDetector()
word = ""
is_running = True

async def detect_signs(websocket):
    global word, is_running

    while is_running:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=False)
        landmarks = detector.find_position(img)

        # Process hand detection
        if landmarks and len(landmarks) == 1:
            lmlist = landmarks[0][1]
            location_vector = np.array([coord for lm in lmlist for coord in lm[1:3]]).reshape(1, -1)

            if model.predict_proba(location_vector).max() > 0.7:
                detected_letter = model.predict(location_vector)[0]
                word += detected_letter
                await websocket.send(f'{{"type": "word", "data": "{word}"}}')

        # Encode frame as base64
        _, buffer = cv2.imencode(".jpg", img)
        frame_data = base64.b64encode(buffer).decode()
        await websocket.send(f'{{"type": "frame", "data": "{frame_data}"}}')

        await asyncio.sleep(0.1)

async def handle_client(websocket):
    global is_running, word

    detect_task = asyncio.create_task(detect_signs(websocket))

    async for message in websocket:
        if message == "stop":
            detect_task.cancel()
            if word:
                play_audio(word)
            word = ""

def play_audio(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

async def main():
    async with websockets.serve(handle_client, "localhost", 5000):
        await asyncio.Future()

asyncio.run(main())
