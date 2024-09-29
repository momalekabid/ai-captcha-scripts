import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
client = ElevenLabs(
    api_key=ELEVENLABS_KEY,
)

def text_to_speech_file(text: str, file_name: str) -> str:
    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id="0m2tDjDewtOfXrhxqgrJ", 
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
    # Using the provided name as the file name for the output MP3 file
    save_file_path = f"{file_name}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path

# Create a dictionary with categories and dialogues
categories_and_dialogues = {
    "test1": "welcome to the humaniator 3000. we’re here to confirm whether you’re a human. for our first test, prove your humanity by lifting your hands above your head in the next 10 seconds.",
    "pass1": "great! you passed",
    "fail1": "a possibility you might be a robot. this is a robot.",
    "test2": "next, we’re going to check your temperature",
    "pass1": "looks good! moving on to the next",
    "fail1": "yea that’s not looking too hot. come back another time!",
    "test3": "whoopsies! that was a surprise. test 3: reflexes.",
    "pass3": "you passed",
    "fail3": "you failed. ",
    "test4": "now the hardest test of them all. close your eyes, what do you smell?",
    "pass4": "that’s right! it’s a pie. congratulations you just passed the whole captcha. enjoy the party",
    "fail4": "yea that's too bad so sad. seems like you might not be human",
}

# Loop through the dictionary to generate audio files for each category
for category, dialogue in categories_and_dialogues.items():
    file_path = text_to_speech_file(dialogue, category)
    print(f"Generated audio file for {category}: {file_path}")






