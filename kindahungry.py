import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import requests
from deepface import DeepFace
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Steps with descriptions
steps = [
    {"name": "Raise Your Arms", "description": "Lift both arms above your head"},
    {"name": "Spin Around", "description": "Turn in a full circle"},
    {"name": "Make a Funny Face", "description": "Show a non-neutral expression"},
]
current_step = 0
game_status = "ongoing"

# Load MoveNet Thunder model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures["serving_default"]

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Keypoint dictionary
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def run_inference(image):
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    keypoints_with_scores = outputs["output_0"].numpy()
    return keypoints_with_scores


def determine_moves(keypoints_with_scores, threshold=0.15):
    keypoints = keypoints_with_scores[0, 0, :, :]
    nose = keypoints[KEYPOINT_DICT["nose"]][:2]
    left_wrist = keypoints[KEYPOINT_DICT["left_wrist"]]
    right_wrist = keypoints[KEYPOINT_DICT["right_wrist"]]
    left_shoulder = keypoints[KEYPOINT_DICT["left_shoulder"]][:2]
    right_shoulder = keypoints[KEYPOINT_DICT["right_shoulder"]][:2]

    hands_up = (
        np.linalg.norm(nose[:2] - left_wrist[:2]) < 0.2 and left_wrist[2] > threshold
    ) or (
        np.linalg.norm(nose[:2] - right_wrist[:2]) < 0.2 and right_wrist[2] > threshold
    )

    sideways = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]) < 0.1

    return hands_up, sideways


def process_frame(frame):
    try:
        predictions = DeepFace.analyze(
            frame, actions=["emotion"], enforce_detection=False
        )

        if predictions:
            emotion_probs = predictions[0]["emotion"]
            neutral_prob = emotion_probs["neutral"]
            non_neutral_prob = 1 - neutral_prob

            return non_neutral_prob

    except Exception as e:
        print(f"Error in processing: {str(e)}")

    return 0  # Return 0 if no face detected or error occurred


def webcam_test():
    global current_step, game_status

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully.")

    hands_raised_count = 0
    HANDS_RAISED_THRESHOLD = 30
    spin_count = 0
    SPIN_THRESHOLD = 1000
    NON_NEUTRAL_THRESHOLD = 0.7

    try:
        while game_status == "ongoing":
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            keypoints_with_scores = run_inference(frame_rgb)
            hands_up, sideways = determine_moves(keypoints_with_scores)

            if current_step == 0:  # Raise hands
                if hands_up:
                    hands_raised_count += 1
                    print(
                        f"Keep hands raised! Progress: {hands_raised_count}/{HANDS_RAISED_THRESHOLD}"
                    )
                else:
                    hands_raised_count = 0

                if hands_raised_count >= HANDS_RAISED_THRESHOLD:
                    current_step = 1
                    requests.post("http://localhost:8000/update/1")

            elif current_step == 1:  # Spin around
                if sideways:
                    spin_count += 1
                print(f"Spin around! Progress: {spin_count}/{SPIN_THRESHOLD}")

                if spin_count >= SPIN_THRESHOLD:
                    current_step = 2
                    requests.post("http://localhost:8000/update/2")

            elif current_step == 2:  # Non-neutral face
                non_neutral_prob = process_frame(frame)
                if non_neutral_prob > NON_NEUTRAL_THRESHOLD:
                    print(
                        f"Non-neutral face detected! Probability: {non_neutral_prob:.2f}"
                    )
                    current_step = 3
                    requests.post("http://localhost:8000/update/3")
                else:
                    print(
                        f"Make a non-neutral face! Current probability: {non_neutral_prob:.2f}"
                    )

            time.sleep(0.1)

    finally:
        cap.release()

    print("Test completed.")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    global current_step, game_status

    if current_step >= len(steps):
        game_status = "complete"
        step_name = "Verification Complete"
        step_description = "You have successfully completed all steps."
    else:
        step_name = steps[current_step]["name"]
        step_description = steps[current_step]["description"]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "current_step": min(current_step, len(steps)),
            "total_steps": len(steps),
            "step_name": step_name,
            "step_description": step_description,
            "game_status": game_status,
        },
    )


@app.post("/update/{step}")
async def update_step(step: int):
    global current_step, game_status
    if 0 <= step <= len(steps):
        current_step = step
        if current_step == len(steps):
            game_status = "complete"
        else:
            game_status = "ongoing"
    return {"status": game_status}


@app.post("/failed")
async def game_failed():
    global game_status
    game_status = "failed"
    return {"status": "failed"}


if __name__ == "__main__":
    # Start the webcam test in a separate thread
    threading.Thread(target=webcam_test, daemon=True).start()

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
