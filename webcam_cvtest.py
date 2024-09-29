import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import requests
from deepface import DeepFace

# load MoveNet Thunder model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

# load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# dictionary for keypoint indices
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def run_inference(image):
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def determine_moves(keypoints_with_scores, threshold=0.15):
    keypoints = keypoints_with_scores[0, 0, :, :]
    nose = keypoints[KEYPOINT_DICT['nose']][:2]
    left_wrist = keypoints[KEYPOINT_DICT['left_wrist']]
    right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]
    left_shoulder = keypoints[KEYPOINT_DICT['left_shoulder']][:2]
    right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']][:2]

    hands_up = (np.linalg.norm(nose[:2] - left_wrist[:2]) < 0.2 and left_wrist[2] > threshold) or \
               (np.linalg.norm(nose[:2] - right_wrist[:2]) < 0.2 and right_wrist[2] > threshold)
    
    sideways = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]) < 0.1

    return hands_up, sideways

def draw_skeleton(image, keypoints_with_scores, threshold=0.3):
    height, width, _ = image.shape
    keypoints = keypoints_with_scores[0, 0, :, :]
    
    # create a copy of the image to draw on
    output_image = image.copy()
    
    # define the connections between keypoints
    connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle')
    ]
    
    # draw the skeleton with a glow effect
    for connection in connections:
        start_idx = KEYPOINT_DICT[connection[0]]
        end_idx = KEYPOINT_DICT[connection[1]]
        
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]
        
        if start_point[2] > threshold and end_point[2] > threshold:
            start_pos = (int(start_point[1] * width), int(start_point[0] * height))
            end_pos = (int(end_point[1] * width), int(end_point[0] * height))
            
            # draw the main line
            cv2.line(output_image, start_pos, end_pos, (0, 255, 0), 2)
            
            # draw the glow effect
            for i in range(1, 3):
                cv2.line(output_image, start_pos, end_pos, (0, 255, 0), 2 + i * 2, cv2.LINE_AA)
    
    # draw keypoints
    for name, idx in KEYPOINT_DICT.items():
        ky, kx, kp_conf = keypoints[idx]
        if kp_conf > threshold:
            cv2.circle(output_image, (int(kx * width), int(ky * height)), 4, (0, 255, 0), -1)
            cv2.circle(output_image, (int(kx * width), int(ky * height)), 6, (0, 255, 0), 1)
    
    return output_image

def process_frame(frame):
    try:
        # analyze the frame
        predictions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        if predictions:
            # get emotion probabilities
            emotion_probs = predictions[0]['emotion']
            neutral_prob = emotion_probs['neutral']
            non_neutral_prob = 1 - neutral_prob
            
            return non_neutral_prob
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
    
    return 0  # Return 0 if no face detected or error occurred

def send_post_request(success):
    print("next")
    # url = "https://example.com/api/captcha-result"  # Replace with your actual API endpoint
    # data = {"success": success}
    # try:
    #     response = requests.post(url, json=data)
    #     response.raise_for_status()
    #     print(f"POST request sent. Success: {success}")
    # except requests.exceptions.RequestException as e:
    #     print(f"Failed to send POST request: {e}")

# init camera
cap = cv2.VideoCapture('/dev/video4')

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")

# Test states
hands_raised_count = 0
HANDS_RAISED_THRESHOLD = 100  # number of frames to keep hands raised
spin_count = 0
test_complete = False
NON_NEUTRAL_THRESHOLD = 0.7  # probability threshold for non-neutral face
non_neutral_face_detected = False

try:
    while not test_complete:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        keypoints_with_scores = run_inference(frame_rgb)
        hands_up, sideways = determine_moves(keypoints_with_scores)

        output_image = draw_skeleton(frame, keypoints_with_scores)

        # Test 1: Raise hands
        if hands_raised_count < HANDS_RAISED_THRESHOLD:
            if hands_up:
                hands_raised_count += 1
                # cv2.putText(output_image, f"Keep hands raised! Progress: {hands_raised_count}/{HANDS_RAISED_THRESHOLD}", (30, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"hands raised: {hands_raised_count}")
            else:
                hands_raised_count = 0
                # cv2.putText(output_image, "Raise your hands!", (30, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Test 2: Spin around (and hold hands up while spinning)
        if spin_count < 100 and hands_up:
            if sideways:
                spin_count += 2 
            # cv2.putText(output_image, f"Spin around! Progress: {spin_count}/100", (30, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"spin count: {spin_count}")
        elif spin_count >= 100:
            time.sleep(5) ## wait here for 5 seconds
            if not non_neutral_face_detected:
                non_neutral_prob = process_frame(frame)
                if non_neutral_prob > NON_NEUTRAL_THRESHOLD:
                    non_neutral_face_detected = True
                    print(f"Non-neutral face detected! Probability: {non_neutral_prob:.2f}")
                    test_complete = True
            else:
                print(f"Make a non-neutral face! Current probability: {non_neutral_prob:.2f}")
        # cv2.imshow('Webcam Test', output_image)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    # Send POST request based on test completion
    # send_post_request(test_complete)

finally:
    cap.release()
    # cv2.destroyAllWindows()