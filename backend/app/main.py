from flask import Flask, render_template, jsonify, Response, request
import cv2
import numpy as np
import time
import os
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from typing import IO

app = Flask(__name__)

# Set your ElevenLabs API key here
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Global variable to store the camera object
camera = None

def generate_frames():
    global camera
    while True:
        if camera is None:
            camera = cv2.VideoCapture(0)
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            

def text_to_speech_stream(text: str) -> IO[bytes]:
    # Perform the text-to-speech conversion
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Return the stream for further use
    return audio_stream

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_captcha', methods=['POST'])
def start_captcha():
    # Placeholder for CAPTCHA logic
    time.sleep(2)  # Simulate some processing time
    return jsonify({"message": "CAPTCHA started"})

@app.route('/get_audio', methods=['POST'])
def get_audio():
    text = request.json['text']
    audio_stream = text_to_speech_stream(text)
    return Response(audio_stream, mimetype='audio/mpeg')

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    text = request.json['text']
    audio = generate(
        text=text,
        voice="Adam",
        model="eleven_multilingual_v2"
    )
    return Response(audio, mimetype='audio/mpeg')

if __name__ == '__main__':
    camera = cv2.VideoCapture(1)
    app.run(host='0.0.0.0', port=5001, debug=True)