import streamlit as st
import random
import json
import os
import re 
from Levenshtein import distance 
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import tempfile 
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment

st.set_page_config(layout="wide", page_title="Quran Game: Random Ayat")

# --- Konfigurasi Azure Speech Service ---
load_dotenv() 
SPEECH_KEY = os.environ.get('AZURE_SPEECH_KEY')
SPEECH_REGION = os.environ.get('AZURE_SPEECH_REGION')

if not SPEECH_KEY or not SPEECH_REGION:
    st.error("Azure Speech Key or Region not set in environment variables.")
    st.info("Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables in your .env file.")
    st.stop() 

def speech_to_text_from_file(file_path):
    """Converts audio from a file to text using Azure Speech SDK."""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "ar-SA" 
    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "Could not recognize speech."
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            return f"Error: {cancellation_details.error_details}"
        else:
            return f"Canceled: {cancellation_details.reason}"
    return "An unknown error occurred."

def enhance_audio(input_path, output_path, target_dBFS=-20.0):
    """
    Enhances audio by reducing noise and normalizing volume.
    1. Noise reduction with noisereduce
    2. Volume normalization with pydub
    """
    # --- Step 1: Noise reduction ---
    y, sr = librosa.load(input_path, sr=None)
    reduced = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced, sr)

    # --- Step 2: Normalize volume ---
    audio = AudioSegment.from_file(output_path)
    change_in_dBFS = target_dBFS - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format="wav")

# --- Fungsi untuk Memuat Data Quran ---
@st.cache_data 
def load_quran_data():
    base_path = "quran/source" 
    all_ayats_data = []

    try:
        with open(os.path.join(base_path, "surah.json"), "r", encoding="utf-8") as f: 
            surah_list_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: File '{os.path.join(base_path, 'surah.json')}' not found. Ensure folder structure is correct.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading surah.json: {e}")
        st.stop()

    for surah_info in surah_list_data:
        surah_number_str = surah_info['index'] 
        surah_number = int(surah_number_str) 
        surah_name_ar = surah_info['titleAr'] 
        surah_ayat_count = surah_info['count'] 

        try:
            arabic_file_path = os.path.join(base_path, "surah", f"surah_{surah_number}.json") 
            with open(arabic_file_path, "r", encoding="utf-8") as f:
                surah_arabic_data = json.load(f)
            
            arabic_verses_map = surah_arabic_data['verse'] 

            for i in range(1, surah_ayat_count + 1): 
                ayat_key = f"verse_{i}"
                if ayat_key in arabic_verses_map:
                    arabic_text = arabic_verses_map[ayat_key]
                else:
                    st.warning(f"Warning: Arabic text for {ayat_key} in Surah {surah_number_str} not found. Skipping this verse.")
                    continue 

                audio_path = os.path.join(base_path, "audio", surah_number_str, f"{str(i).zfill(3)}.mp3") 
                
                all_ayats_data.append({
                    "surah_number": surah_number,
                    "surah_name_ar": surah_name_ar, 
                    "ayat_number": i,
                    "text_ar": arabic_text,
                    "audio_path": audio_path 
                })
        except FileNotFoundError as e:
            st.warning(f"Warning: Arabic text file for Surah {surah_number_str} not found ({e}). Continuing to next surah.")
            continue
        except KeyError as e:
            st.warning(f"Warning: JSON structure for Surah {surah_number_str} does not match expectations (key {e} missing). Continuing to next surah.")
            continue
        except Exception as e:
            st.warning(f"Warning: Failed to load data for Surah {surah_number_str} ({e}). Continuing to next surah.")
            continue
            
    return all_ayats_data

# --- Fungsi untuk Memuat Data Juz ---
@st.cache_data 
def load_juz_data():
    base_path = "quran/source" 
    juz_file_path = os.path.join(base_path, "juz.json") 
    juz_data_map = {}
    try:
        with open(juz_file_path, "r", encoding="utf-8") as f: 
            juz_list = json.load(f) 
            for juz_info in juz_list: 
                juz_index = int(juz_info['index']) 
                start_info = juz_info['start'] 
                end_info = juz_info['end'] 
                start_surah = int(start_info['index']) 
                start_ayat = int(start_info['verse'].replace('verse_', '')) 
                end_surah = int(end_info['index']) 
                end_ayat = int(end_info['verse'].replace('verse_', '')) 
                
                juz_data_map[juz_index] = {
                    'start_surah': start_surah,
                    'start_ayat': start_ayat,
                    'end_surah': end_surah,
                    'end_ayat': end_ayat
                }
    except FileNotFoundError:
        st.error(f"Error: File '{juz_file_path}' not found. Ensure folder structure is correct.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading juz.json: {e}")
        st.stop()
    return juz_data_map

# Memuat semua data Quran dan Juz saat aplikasi dimulai
full_quran_data = load_quran_data()
juz_ranges_map = load_juz_data()

if not full_quran_data:
    st.error("No Quran data loaded successfully. Ensure JSON files exist and are correctly formatted.")
    st.stop()

# --- Fuzzy Matching Functions ---
def normalize_arabic_text(text):
    """
    Normalizes Arabic text by removing diacritics and normalizing certain characters.
    """
    text = re.sub(r'[\u064B-\u0652\u0670\u06D6-\u06DC\u06DF-\u06E4\u06EA-\u06ED]', '', text) 
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ٰ', 'ا')
    text = text.replace('ـ', '')
    text = text.replace('لأ', 'لا').replace('لإ', 'لا').replace('لآ', 'لا')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_ayat_in_juz(ayat_data, juz_number, juz_ranges_map_param):
    """Checks if a given ayat falls within the specified juz range using the loaded map."""
    juz_range = juz_ranges_map_param.get(juz_number)
    if not juz_range:
        return False
    s_surah, s_ayat = juz_range['start_surah'], juz_range['start_ayat']
    e_surah, e_ayat = juz_range['end_surah'], juz_range['end_ayat']
    ayat_surah, ayat_ayat = ayat_data['surah_number'], ayat_data['ayat_number']
    if ayat_surah > s_surah and ayat_surah < e_surah: return True
    elif ayat_surah == s_surah and ayat_ayat >= s_ayat: return True
    elif ayat_surah == e_surah and ayat_ayat <= e_ayat: return True
    return False

@st.cache_data
def filter_quran_by_juz(full_quran_data_list, selected_juz_options, juz_ranges_map_param):
    """Filters the full Quran data list based on selected Juz options."""
    if "30 Juz" in selected_juz_options or not selected_juz_options:
        return full_quran_data_list
    filtered_data = []
    selected_juz_numbers = [int(opt.split(" ")[1]) for opt in selected_juz_options if opt.startswith("Juz ")]
    if not selected_juz_numbers: return full_quran_data_list
    for ayat_data in full_quran_data_list:
        if any(is_ayat_in_juz(ayat_data, juz_num, juz_ranges_map_param) for juz_num in selected_juz_numbers):
            filtered_data.append(ayat_data)
    return filtered_data

def get_next_ayat(current_ayat_info, all_quran_data_list): 
    """Finds the next verse in the given list."""
    current_index_in_list = -1
    for i, ayat_data in enumerate(all_quran_data_list):
        if ayat_data['surah_number'] == current_ayat_info['surah_number'] and ayat_data['ayat_number'] == current_ayat_info['ayat_number']:
            current_index_in_list = i
            break
    if current_index_in_list == -1 or current_index_in_list + 1 >= len(all_quran_data_list): return None, -1 
    next_ayat_candidate = all_quran_data_list[current_index_in_list + 1]
    if (next_ayat_candidate['surah_number'] > current_ayat_info['surah_number']) or \
       (next_ayat_candidate['surah_number'] == current_ayat_info['surah_number'] and next_ayat_candidate['ayat_number'] == current_ayat_info['ayat_number'] + 1):
        return next_ayat_candidate, current_index_in_list + 1
    return None, -1

def fuzzy_match_arabic_ayat(original_text_ar, transcribed_text_ar, threshold=0.7):
    """
    Compares two Arabic texts using Levenshtein distance after normalization.
    """
    normalized_original = normalize_arabic_text(original_text_ar)
    normalized_transcribed = normalize_arabic_text(transcribed_text_ar)
    if not normalized_original: return False, 0.0
    lev_distance = distance(normalized_original, normalized_transcribed)
    similarity_score = (len(normalized_original) - lev_distance) / len(normalized_original)
    return similarity_score >= threshold, similarity_score

# --- Inisialisasi State Aplikasi ---
if 'current_ayat_index' not in st.session_state:
    st.session_state.current_ayat_index = 0 
if 'audio_input_key' not in st.session_state:
    st.session_state.audio_input_key = 0
if 'last_transcription' not in st.session_state:
    st.session_state.last_transcription = ""
if 'last_match_result' not in st.session_state:
    st.session_state.last_match_result = None
if 'previous_juz_options' not in st.session_state:
    st.session_state.previous_juz_options = ["30 Juz"]

st.markdown("# 📖 Quran Game")
st.markdown("Tags : :blue-badge[Speech Recognition], :blue-badge[Speech-to-text], :blue-badge[Fuzzy Analysis]")

with st.expander("📝 Readme", expanded=True):
    st.markdown("""
    **Quran Game** is an educational game that tests our ability to correctly continue reciting Quran verses. 
    This game uses Speech-to-Text technology to recognize your recitation and compare it with the target verse.
    
    **How to Play:**
    1. Select the Juz you want to play.
    2. Recite the displayed verse.
    3. Record your recitation using the microphone.
    4. The system will tell you if your recitation is correct (matches the target verse).
    
    **Features:**
    - Flexible Juz selection.
    - Listen verse recitation.
    - Option to change to another verse.
    
    Enjoy and hope it's helpful!!
    """)

options = st.multiselect(
    "Select Juz to play",
    ["30 Juz"] + [f"Juz {i}" for i in range(1, 31)], 
    default=st.session_state.previous_juz_options,
    key="juz_selector" 
)

filtered_quran_data = filter_quran_by_juz(full_quran_data, options, juz_ranges_map)

if not filtered_quran_data:
    st.warning("No verses found for the selected Juz options. Please select other Juz.")
    st.stop()

if st.session_state.get('previous_juz_options') != options:
    st.session_state.current_ayat_index = random.randint(0, len(filtered_quran_data) - 1)
    st.session_state.previous_juz_options = options 
    st.session_state.audio_input_key = 0
    st.session_state.last_transcription = ""
    st.session_state.last_match_result = None
    st.rerun() 

current_ayat = filtered_quran_data[st.session_state.current_ayat_index]
next_ayat, next_ayat_index = get_next_ayat(current_ayat, filtered_quran_data)

st.markdown(f"### Clue, verse: **{current_ayat['ayat_number']}**")
st.markdown(f"<h1 style='text-align: right; font-size: 3em; direction: rtl;'>{current_ayat['text_ar']}</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Listen 🔊", use_container_width=True):
        if os.path.exists(current_ayat['audio_path']):
            st.audio(current_ayat['audio_path'], format="audio/mp3", start_time=0, autoplay=True)
        else:
            st.warning(f"Audio file not found: {current_ayat['audio_path']}. Cannot play.")

with col2:
    if st.button("Change Verse ➡️", use_container_width=True):
        new_index = random.randint(0, len(filtered_quran_data) - 1) 
        if len(filtered_quran_data) > 1:
            while new_index == st.session_state.current_ayat_index:
                new_index = random.randint(0, len(filtered_quran_data) - 1)
        st.session_state.current_ayat_index = new_index
        st.session_state.audio_input_key += 1 
        st.session_state.last_transcription = "" 
        st.session_state.last_match_result = None 
        st.rerun()

st.markdown("---")

# Menggunakan st.audio_input untuk merekam audio
uploaded_audio = st.audio_input("Record Recitation 🎤", key=f"audio_input_{st.session_state.audio_input_key}")

if uploaded_audio is not None:
    if next_ayat is None:
        st.warning("No next verse to connect. Recording cannot be processed.")
    else:
        temp_audio_file_path = None
        try:
            # Dapatkan format dari Streamlit
            audio_format = uploaded_audio.type.split('/')[-1] if uploaded_audio.type else 'wav'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as temp_audio_file:
                temp_audio_file.write(uploaded_audio.read())
                temp_audio_file_path = temp_audio_file.name

            st.write("Audio recorded successfully!")
            
            info_placeholder = st.empty()
            info_placeholder.info("Analyzing your audio for transcription...")
            
            # --- Enhance audio before sending to Azure ---
            enhanced_audio_path = temp_audio_file_path.replace('.wav', '_enhanced.wav')
            enhance_audio(temp_audio_file_path, enhanced_audio_path)

            # --- Speech recognition on enhanced audio ---
            transcribed_text = speech_to_text_from_file(enhanced_audio_path)

            st.session_state.last_transcription = transcribed_text

            is_match, similarity_score = fuzzy_match_arabic_ayat(next_ayat['text_ar'], transcribed_text)
            st.session_state.last_match_result = (is_match, similarity_score)
            info_placeholder.empty()
            
            st.success("Transcription complete!")
    
            if is_match:
                st.success(f"**🥳 Match! You successfully connected the verse!** (Similarity Score: {similarity_score:.2f})")
            else:
                st.error(f"**😔 No match. Please try again.** (Similarity Score: {similarity_score:.2f})")
                st.info("Ensure your audio contains the correct recitation of the target verse.")

            with st.expander("See Transcription Result", expanded=True):
                st.markdown(f"**Your Transcription:** {transcribed_text}")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        finally:
            if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)
            if 'enhanced_audio_path' in locals() and os.path.exists(enhanced_audio_path):
                os.remove(enhanced_audio_path)


st.markdown("---")
with st.expander("Answer Key", icon="💡", expanded=False):
    if next_ayat:
        st.markdown(f"### Surah: **{next_ayat['surah_name_ar']}**, Verse: **{next_ayat['ayat_number']}**")
        st.markdown(f"<h3 style='text-align: right; font-size: 2em; direction: rtl;'>{next_ayat['text_ar']}</h3>", unsafe_allow_html=True)
    else:
        st.warning("No target verse available. You have reached the end of the Quran.")