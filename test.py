import streamlit as st
from google.cloud import texttospeech
import os

def synthesize_text(text):
    """Synthesizes speech from the input string of text and returns the audio file path."""
    # Ensure your environment variables are set for Google Cloud authentication
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-C",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # Write the audio content to an MP3 file
    audio_file_path = "output.mp3"
    with open(audio_file_path, "wb") as out:
        out.write(response.audio_content)
    
    return audio_file_path

def main():
    st.title('Text to Speech Synthesis')
    user_input = st.text_area("Enter text here:", "Hello there, welcome to our text to speech app.")
    
    if st.button("Synthesize"):
        audio_file_path = synthesize_text(user_input)
        audio_file = open(audio_file_path, "rb")
        st.audio(audio_file.read(), format="audio/mp3")

if __name__ == "__main__":
    main()
