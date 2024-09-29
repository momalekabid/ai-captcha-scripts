import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

# load MoveNet Thunder model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

# dictionary for keypoint indices
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def run_inference(image):
    # resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)  # thunder model uses 256x256
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    # run model inference.
    outputs = movenet(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def determine_moves(keypoints_with_scores, threshold=0.2):
    keypoints = keypoints_with_scores[0, 0, :, :]
    
    detected_moves = []
    nose = keypoints[KEYPOINT_DICT['nose']][:2]
    left_shoulder = keypoints[KEYPOINT_DICT['left_shoulder']][:2]
    right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']][:2]
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
    nose = keypoints[KEYPOINT_DICT['nose']]
    left_wrist = keypoints[KEYPOINT_DICT['left_wrist']]
    right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]


    left_ankle = keypoints[KEYPOINT_DICT['left_ankle']]
    right_ankle = keypoints[KEYPOINT_DICT['right_ankle']]
    left_knee = keypoints[KEYPOINT_DICT['left_knee']]
    right_knee = keypoints[KEYPOINT_DICT['right_knee']]
    
    if left_ankle[2] > threshold and left_knee[2] > threshold and (right_ankle[2] < threshold or right_knee[2] < threshold):
        detected_moves.append("Standing on (right) Leg")
    elif right_ankle[2] > threshold and right_knee[2] > threshold and (left_ankle[2] < threshold or left_knee[2] < threshold):
        detected_moves.append("Standing on (left) Leg")
    # hopping



    if shoulder_distance <= 0.1:  # shoulders are horizontally close together,
        ## TODO: add a check for if the person made a full 360 turn or not
        detected_moves.append("Spinning")
    elif (np.linalg.norm(nose[:2] - left_wrist[:2]) < 0.2 and left_wrist[2] > threshold) or \
       (np.linalg.norm(nose[:2] - right_wrist[:2]) < 0.2 and right_wrist[2] > threshold):
        detected_moves.append("Hands Above Head")
    return detected_moves if detected_moves else ["No Move Detected"]

# added these variables for move display fine-grained control
last_move_time = time.time()
current_moves = ["No Move Detected"]

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

# initialize webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")


try:
    while True:
        # capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # run pose estimation
        keypoints_with_scores = run_inference(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        moves = determine_moves(keypoints_with_scores)

        # update moves if new ones are detected or if 1 seconds have passed
        current_time = time.time()
        if moves != ["No Move Detected"] or current_time - last_move_time >= 1:
            current_moves = moves
            last_move_time = current_time

        # draw cool skeleton on the image
        output_image = draw_skeleton(frame, keypoints_with_scores)

        # display moves
        for i, move in enumerate(current_moves):
            cv2.putText(output_image, f"Move: {move}", (30, 50 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # show images
        cv2.imshow('Webcam MoveNet Outline', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()