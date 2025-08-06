import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pydub
from azure.cognitiveservices.speech import SpeechConfig
from azure.cognitiveservices.speech.audio import AudioConfig, PushAudioInputStream
from azure.cognitiveservices.speech.transcription import ConversationTranscriber, ConversationTranscriptionEventArgs
import os
import threading
import time
import logging
from dotenv import load_dotenv

# Logging untuk debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION")

# Fungsi transkripsi real-time
def process_webrtc_transcription(webrtc_ctx):
    push_stream = PushAudioInputStream()
    audio_config = AudioConfig(stream=push_stream)

    speech_config = SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "id-ID"
    transcriber = ConversationTranscriber(speech_config, audio_config)

    results = []

    def on_transcribed(evt: ConversationTranscriptionEventArgs):
        if evt.result.text:
            results.append(evt.result.text)
            st.session_state.transcription = "\n".join(results[-20:])  # Batasi agar tidak terlalu panjang
            logger.info(f"Transcribed: {evt.result.text}")

    transcriber.transcribed.connect(on_transcribed)
    transcriber.start_transcribing_async()
    logger.info("Transcription started...")

    try:
        while webrtc_ctx.state.playing:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if not audio_frames:
                continue

            frame = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                frame += sound

            if len(frame) > 0:
                frame = frame.set_channels(1).set_frame_rate(16000)
                push_stream.write(frame.raw_data)

            time.sleep(0.1)

    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        transcriber.stop_transcribing_async()
        push_stream.close()
        logger.info("Transcription stopped.")

# UI Streamlit
st.title("🎙️ Real-time Speech-to-Text Quran Game")

# Inisialisasi session state
if "transcription" not in st.session_state:
    st.session_state.transcription = ""

# WebRTC Audio Streamer
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"video": False, "audio": True},
)

# UI: Saat streaming aktif
if webrtc_ctx.state.playing:
    st.subheader("📜 Hasil Transkripsi:")
    placeholder = st.empty()

    # Jalankan transkripsi di thread terpisah sekali saja
    if "transcription_thread" not in st.session_state:
        thread = threading.Thread(target=process_webrtc_transcription, args=(webrtc_ctx,))
        thread.start()
        st.session_state.transcription_thread = thread

    # Tampilkan teks transkripsi real-time
    while webrtc_ctx.state.playing:
        placeholder.markdown(st.session_state.transcription)
        time.sleep(0.5)

# UI: Saat streaming tidak aktif
else:
    st.info("Klik 'Start' untuk memulai transkripsi audio.")
    if "transcription_thread" in st.session_state:
        del st.session_state["transcription_thread"]
