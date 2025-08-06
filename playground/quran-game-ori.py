import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import os

chunks = []

load_dotenv()
subscription_key = os.environ.get("AZURE_SPEECH_KEY")
service_region = os.environ.get("AZURE_SPEECH_REGION")
speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)
audio_config = speechsdk.AudioConfig(use_default_microphone=True)

def recognized_callback(evt):
    global chunks
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        chunks.append(evt.result.text)
        print(f"Recognized: {evt.result.text}")

def main():
    st.title("Azure Speech Service with Streamlit")
    
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    recognizer.recognized.connect(recognized_callback)
    recognizer.start_continuous_recognition()
    
    st.write("Speak into your microphone.")
    while True:
        if chunks:
            st.write(chunks)
            chunks.clear()
            st.write("Done")

if __name__ == "__main__":
    main()