# pip install opencv-python mediapipe pyautogui SpeechRecognition pyttsx3 numpy pyaudio
# Optional emotion engines:
# pip install deepface
# pip install fer tensorflow

import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyttsx3
import speech_recognition as sr


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01


CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
TARGET_FPS = 30

CURSOR_SMOOTHING = 7.0
CLICK_DISTANCE_RATIO = 0.045
CLICK_COOLDOWN = 0.45
DRAG_HOLD_SECONDS = 0.45
SCROLL_COOLDOWN = 0.10
GESTURE_SPEAK_COOLDOWN = 2.5
VOICE_SLEEP_SECONDS = 0.1

DRAW_COLOR = (0, 255, 180)
DRAW_THICKNESS = 7


@dataclass
class SharedState:
    running: bool = True
    voice_status: str = "Voice: ready"
    active_mode: str = "Mouse"
    drawing_enabled: bool = False
    last_voice_command: str = ""


class Speaker:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def say(self, text):
        if text:
            self.queue.put(text)

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 0.9)
        except Exception:
            engine = None

        while True:
            text = self.queue.get()
            if text is None:
                break
            if engine is None:
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

    def stop(self):
        self.queue.put(None)


class VoiceAssistant:
    def __init__(self, state, speaker):
        self.state = state
        self.speaker = speaker
        self.recognizer = sr.Recognizer()
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)

    def start(self):
        self.thread.start()

    def _listen_once(self, source, language):
        try:
            audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
            return self.recognizer.recognize_google(audio, language=language).lower().strip()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            self.state.voice_status = "Voice: network/API error"
            return ""
        except Exception:
            self.state.voice_status = "Voice: microphone error"
            return ""

    def _listen_loop(self):
        try:
            mic = sr.Microphone()
        except Exception:
            self.state.voice_status = "Voice: microphone unavailable"
            return

        with mic as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.7)
            except Exception:
                pass

            while self.state.running:
                self.state.voice_status = "Voice: listening"
                command = self._listen_once(source, "en-IN")
                if not command:
                    command = self._listen_once(source, "ta-IN")
                if command:
                    self._handle_command(command)
                time.sleep(VOICE_SLEEP_SECONDS)

    def _handle_command(self, command):
        self.state.last_voice_command = command
        self.state.voice_status = f"Voice: {command[:32]}"

        if "click" in command or "கிளிக்" in command:
            pyautogui.click()
            self.speaker.say("Click")
        elif "open chrome" in command or "chrome open" in command or "குரோம் திற" in command:
            self._open_chrome()
            self.speaker.say("Opening Chrome")
        elif "scroll down" in command:
            pyautogui.scroll(-6)
            self.speaker.say("Scroll down")
        elif "screenshot" in command:
            self._screenshot()
            self.speaker.say("Screenshot saved")
        elif "stop" in command or "நிறுத்து" in command:
            self.state.running = False
            self.speaker.say("Stopping")

    def _open_chrome(self):
        try:
            if os.name == "nt":
                subprocess.Popen(["cmd", "/c", "start", "chrome"], shell=False)
            elif os.name == "posix":
                subprocess.Popen(["google-chrome"])
        except Exception:
            try:
                subprocess.Popen(["chrome"])
            except Exception:
                self.state.voice_status = "Voice: Chrome not found"

    def _screenshot(self):
        try:
            folder = os.path.join(os.getcwd(), "screenshots")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            pyautogui.screenshot(path)
        except Exception:
            self.state.voice_status = "Voice: screenshot failed"


class EmotionDetector:
    def __init__(self):
        self.backend = "haar"
        self.deepface = None
        self.fer_detector = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.last_emotion = "Neutral"
        self.last_boxes = []
        self.last_run = 0

        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.backend = "deepface"
        except Exception:
            try:
                from fer import FER
                self.fer_detector = FER(mtcnn=False)
                self.backend = "fer"
            except Exception:
                self.backend = "haar"

    def detect(self, frame):
        now = time.time()
        if now - self.last_run < 0.35:
            return self.last_boxes, self.last_emotion

        self.last_run = now
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(45, 45))
        boxes = [(x * 2, y * 2, w * 2, h * 2) for x, y, w, h in faces[:1]]

        emotion = self.last_emotion
        if boxes:
            x, y, w, h = boxes[0]
            face = frame[max(0, y):y + h, max(0, x):x + w]
            emotion = self._predict_emotion(face)

        self.last_boxes = boxes
        self.last_emotion = emotion
        return boxes, emotion

    def _predict_emotion(self, face):
        if face.size == 0:
            return "Neutral"

        if self.deepface is not None:
            try:
                result = self.deepface.analyze(face, actions=["emotion"], enforce_detection=False, silent=True)
                if isinstance(result, list):
                    result = result[0]
                return self._map_emotion(result.get("dominant_emotion", "neutral"))
            except Exception:
                pass

        if self.fer_detector is not None:
            try:
                result = self.fer_detector.top_emotion(face)
                if result and result[0]:
                    return self._map_emotion(result[0])
            except Exception:
                pass

        return self._simple_emotion_guess(face)

    def _simple_emotion_guess(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        if brightness > 125 and contrast > 48:
            return "Happy"
        if brightness < 75:
            return "Sad"
        if contrast > 62:
            return "Angry"
        return "Neutral"

    def _map_emotion(self, emotion):
        emotion = str(emotion).lower()
        if emotion in ("happy", "surprise"):
            return "Happy"
        if emotion in ("sad", "fear", "disgust"):
            return "Sad"
        if emotion == "angry":
            return "Angry"
        return "Neutral"


class HandGestureSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_cursor_x = self.screen_w / 2
        self.prev_cursor_y = self.screen_h / 2
        self.last_click = 0
        self.click_start = None
        self.dragging = False
        self.last_scroll = 0
        self.last_spoken_gesture = ""
        self.last_spoken_time = 0
        self.prev_draw_point = None
        self.canvas = None

    def process(self, frame, state, speaker):
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        gesture = "No Hand"
        landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
            landmarks = self._landmark_points(hand_landmarks, frame.shape)
            gesture = self._classify_gesture(landmarks)
            self._handle_mouse(landmarks, gesture)
            self._handle_drawing(frame, landmarks, gesture, state)
            self._speak_gesture(gesture, speaker)
        else:
            self._end_drag()
            self.prev_draw_point = None

        blended = cv2.addWeighted(frame, 1.0, self.canvas, 0.85, 0)
        return blended, gesture, landmarks

    def _landmark_points(self, hand_landmarks, shape):
        h, w = shape[:2]
        return [(int(lm.x * w), int(lm.y * h), lm.z) for lm in hand_landmarks.landmark]

    def _distance(self, points, a, b):
        return math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1])

    def _finger_states(self, points):
        wrist_x = points[0][0]
        thumb_up = points[4][0] < points[3][0] if points[4][0] < wrist_x else points[4][0] > points[3][0]
        index_up = points[8][1] < points[6][1]
        middle_up = points[12][1] < points[10][1]
        ring_up = points[16][1] < points[14][1]
        pinky_up = points[20][1] < points[18][1]
        return thumb_up, index_up, middle_up, ring_up, pinky_up

    def _classify_gesture(self, points):
        thumb, index, middle, ring, pinky = self._finger_states(points)
        states = [thumb, index, middle, ring, pinky]
        palm_size = max(1.0, self._distance(points, 0, 9))
        ok_distance = self._distance(points, 4, 8) / palm_size

        if ok_distance < 0.34 and middle and ring and pinky:
            return "OK Sign"
        if states == [True, False, False, False, False]:
            return "Thumbs Up"
        if states == [False, True, True, False, False]:
            return "Peace"
        if all(states):
            return "Open Palm"
        if not any(states):
            return "Fist"
        if states == [False, True, False, False, False] or states == [True, True, False, False, False]:
            return "Pointing"
        if states == [True, True, False, False, True] or states == [False, True, False, False, True]:
            return "Rock Sign"
        return "Hand"

    def _handle_mouse(self, points, gesture):
        index_x, index_y, _ = points[8]
        frame_h = max(1, CAMERA_HEIGHT)
        frame_w = max(1, CAMERA_WIDTH)
        target_x = np.interp(index_x, (70, frame_w - 70), (0, self.screen_w))
        target_y = np.interp(index_y, (70, frame_h - 70), (0, self.screen_h))
        cursor_x = self.prev_cursor_x + (target_x - self.prev_cursor_x) / CURSOR_SMOOTHING
        cursor_y = self.prev_cursor_y + (target_y - self.prev_cursor_y) / CURSOR_SMOOTHING
        self.prev_cursor_x, self.prev_cursor_y = cursor_x, cursor_y

        if gesture in ("Pointing", "Peace", "OK Sign", "Hand"):
            try:
                pyautogui.moveTo(cursor_x, cursor_y)
            except Exception:
                pass

        palm_size = max(1.0, self._distance(points, 0, 9))
        click_distance = self._distance(points, 8, 12) / palm_size
        now = time.time()

        if gesture == "Peace" and click_distance < CLICK_DISTANCE_RATIO * 5:
            if self.click_start is None:
                self.click_start = now
            if now - self.click_start > DRAG_HOLD_SECONDS and not self.dragging:
                try:
                    pyautogui.mouseDown()
                    self.dragging = True
                except Exception:
                    pass
            elif now - self.last_click > CLICK_COOLDOWN and not self.dragging:
                try:
                    pyautogui.click()
                    self.last_click = now
                except Exception:
                    pass
        else:
            self.click_start = None
            self._end_drag()

        thumb, index, middle, ring, pinky = self._finger_states(points)
        if index and middle and ring and pinky and not thumb and now - self.last_scroll > SCROLL_COOLDOWN:
            delta = points[8][1] - points[5][1]
            try:
                pyautogui.scroll(-4 if delta > 0 else 4)
                self.last_scroll = now
            except Exception:
                pass

    def _end_drag(self):
        if self.dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        self.dragging = False

    def _handle_drawing(self, frame, points, gesture, state):
        if gesture == "Fist":
            self.canvas[:] = 0
            self.prev_draw_point = None
            state.drawing_enabled = False
            state.active_mode = "Canvas cleared"
            return

        if gesture == "Open Palm":
            self.prev_draw_point = None
            state.drawing_enabled = False
            state.active_mode = "Mouse"
            return

        if gesture == "Pointing":
            state.drawing_enabled = True
            state.active_mode = "Air Drawing"
            x, y, _ = points[8]
            current = (x, y)
            if self.prev_draw_point is not None:
                cv2.line(self.canvas, self.prev_draw_point, current, DRAW_COLOR, DRAW_THICKNESS, cv2.LINE_AA)
            self.prev_draw_point = current
            cv2.circle(frame, current, 8, DRAW_COLOR, -1)
        else:
            self.prev_draw_point = None
            if not self.dragging:
                state.active_mode = "Mouse"

    def _speak_gesture(self, gesture, speaker):
        if gesture in ("No Hand", "Hand"):
            return
        now = time.time()
        if gesture != self.last_spoken_gesture or now - self.last_spoken_time > GESTURE_SPEAK_COOLDOWN:
            self.last_spoken_gesture = gesture
            self.last_spoken_time = now
            speaker.say(gesture)


def draw_overlay(frame, fps, state, gesture, emotion, face_boxes):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (360, 150), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

    lines = [
        f"FPS: {fps:.1f}",
        f"Mode: {state.active_mode}",
        f"Gesture: {gesture}",
        f"Emotion: {emotion}",
        state.voice_status,
    ]

    y = 38
    for text in lines:
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 25

    if state.last_voice_command:
        cv2.putText(
            frame,
            f"Last command: {state.last_voice_command[:42]}",
            (24, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (180, 255, 220),
            2,
            cv2.LINE_AA,
        )

    for x, y, bw, bh in face_boxes:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 255), 2)
        cv2.rectangle(frame, (x, max(0, y - 30)), (x + 160, y), (0, 220, 255), -1)
        cv2.putText(frame, emotion, (x + 8, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, "Press Q to quit", (w - 180, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    state = SharedState()
    speaker = Speaker()
    voice = VoiceAssistant(state, speaker)
    hand_system = HandGestureSystem()
    emotion_detector = EmotionDetector()

    cap = open_camera()
    if cap is None:
        print("Webcam error: camera could not be opened.")
        speaker.say("Camera error")
        time.sleep(1)
        speaker.stop()
        return

    voice.start()
    speaker.say("AI virtual mouse started")

    fps = 0.0
    last_time = time.time()
    frame_delay = 1.0 / TARGET_FPS

    try:
        while state.running:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                state.voice_status = "Camera: frame read failed"
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
            frame = cv2.flip(frame, 1)

            frame, gesture, _ = hand_system.process(frame, state, speaker)
            face_boxes, emotion = emotion_detector.detect(frame)

            now = time.time()
            instant_fps = 1.0 / max(0.001, now - last_time)
            fps = fps * 0.88 + instant_fps * 0.12
            last_time = now

            draw_overlay(frame, fps, state, gesture, emotion, face_boxes)
            cv2.imshow("AI Virtual Mouse + Gesture + Air Drawing + Voice + Emotion", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                state.running = False
                break

            elapsed = time.time() - loop_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
    except KeyboardInterrupt:
        state.running = False
    finally:
        state.running = False
        hand_system._end_drag()
        cap.release()
        cv2.destroyAllWindows()
        speaker.say("Stopped")
        time.sleep(0.2)
        speaker.stop()


if __name__ == "__main__":
    main()
