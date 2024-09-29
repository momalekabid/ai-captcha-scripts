import cv2
from deepface import DeepFace
import time

# load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
    try:
        # convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # if faces are detected
        if len(faces) > 0:
            # analyze the frame
            predictions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # draw rectangle around each face and display emotion
            for (x, y, w, h), prediction in zip(faces, predictions):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # get emotion probabilities
                emotion_probs = prediction['emotion']
                dominant_emotion = max(emotion_probs, key=emotion_probs.get)
                
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
    
    return frame

def main():
    # init camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera opened successfully.")

    # lower res 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process the frame
            output_image = process_frame(frame)

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(output_image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show image
            cv2.imshow('DeepFace Emotion Recognition', output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
