# import os
# from typing import IO
# from io import BytesIO
# from elevenlabs import VoiceSettings
# from elevenlabs.client import ElevenLabs
# import sounddevice as sd
# import numpy as np
# from pydub import AudioSegment
# import tempfile

# ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
# client = ElevenLabs(
#     api_key=ELEVENLABS_KEY,
# )

# def text_to_speech_stream(text: str) -> IO[bytes]:
#     # Perform the text-to-speech conversion
#     response = client.text_to_speech.convert(
#         voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
#         output_format="mp3_22050_32",
#         text=text,
#         model_id="eleven_multilingual_v2",
#         voice_settings=VoiceSettings(
#             stability=0.0,
#             similarity_boost=1.0,
#             style=0.0,
#             use_speaker_boost=True,
#         ),
#     )

#     # Create a BytesIO object to hold the audio data in memory
#     audio_stream = BytesIO()

#     # Write each chunk of audio data to the stream
#     for chunk in response:
#         if chunk:
#             audio_stream.write(chunk)

#     # Reset stream position to the beginning
#     audio_stream.seek(0)

#     # Return the stream for further use
#     return audio_stream


# def play_audio_stream(audio_stream: IO[bytes]):
#     # Create a temporary file to store the MP3 data
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
#         temp_file.write(audio_stream.read())
#         temp_file_path = temp_file.name

#     # Load the MP3 file using pydub
#     audio = AudioSegment.from_mp3(temp_file_path)

#     # Convert to numpy array
#     samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

#     # Play the audio
#     sd.play(samples, samplerate=audio.frame_rate)
#     sd.wait()

#     # Clean up the temporary file
#     os.unlink(temp_file_path)

# if __name__ == "__main__":
#     text = "Hello, this is a test of the text-to-speech function."
#     audio_stream = text_to_speech_stream(text)
#     play_audio_stream(audio_stream)


import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
client = ElevenLabs(
    api_key=ELEVENLABS_KEY,
)


def text_to_speech_file(text: str) -> str:
    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id="jBpfuIE2acCO8z3wKNLl", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # uncomment the line below to play the audio back
    # play(response)

    # Generating a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path

text_to_speech_file("testing")



