import cv2
import numpy as np
import requests
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


def process_frame(frame):
    # Define the region of interest (ROI) for the temperature reading
    # Adjust these values based on where the temperature is displayed in your frame
    x, y, w, h = 100, 100, 200, 100  # Example values
    roi = frame[y:y+h, x:x+w]

    # Preprocess the ROI
    _, encoded_image = cv2.imencode('.jpg', roi)
    base64_image = base64.b64encode(encoded_image).decode('ascii')
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What number is shown on the screen?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    
    if response.choices:
        text = response.choices[0].message.content
        # Extract temperature from text
        try:
            temperature = float(text.split()[0])
            return temperature, frame
        except ValueError:
            print("Failed to extract temperature from:", text)
            return None, frame

    return None, frame 
# Initialize video capture
cap = cv2.VideoCapture(1) 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    temperature, frame = process_frame(frame)
    
    if temperature is not None:
        if temperature > 20:
            print(f"Temperature {temperature:.1f}°C is above 20°C")
        else:
            print(f"Temperature {temperature:.1f}°C is not above 20°C")
    
    # Display the frame with bounding box
    cv2.imshow('Thermometer', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()