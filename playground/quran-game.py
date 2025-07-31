import streamlit as st
import random
import json
import os
import re 
from Levenshtein import distance 
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import tempfile # Diperlukan untuk menyimpan file audio sementara

st.set_page_config(layout="wide", page_title="Quran Game: Random Ayat")

# --- Konfigurasi Azure Speech Service ---
load_dotenv() 
SPEECH_KEY = os.environ.get('AZURE_SPEECH_KEY')
SPEECH_REGION = os.environ.get('AZURE_SPEECH_REGION')

if not SPEECH_KEY or not SPEECH_REGION:
    st.error("Azure Speech Key or Region not set in environment variables.")
    st.info("Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables in your .env file.")
    st.stop() 

def speech_to_text_from_mic():
    """Mengambil audio dari mikrofon dan mengonversinya ke teks."""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "ar-SA" 

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
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

# --- Fungsi untuk Memuat Data Quran ---
@st.cache_data 
def load_quran_data():
    base_path = "quran/source" #
    
    all_ayats_data = []

    # 1. Memuat surah.json untuk mendapatkan daftar surah dan metadata (count, title, titleAr)
    try:
        with open(os.path.join(base_path, "surah.json"), "r", encoding="utf-8") as f: #
            surah_list_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: File '{os.path.join(base_path, 'surah.json')}' not found. Ensure folder structure is correct.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading surah.json: {e}")
        st.stop()

    # 2. Iterasi melalui setiap surah untuk memuat teks ayat dan mengumpulkan info
    for surah_info in surah_list_data:
        surah_number_str = surah_info['index'] #
        surah_number = int(surah_number_str) 
        surah_name_ar = surah_info['titleAr'] #
        surah_ayat_count = surah_info['count'] #

        try:
            # Memuat file JSON untuk teks Arab dari surah spesifik
            arabic_file_path = os.path.join(base_path, "surah", f"surah_{surah_number}.json") #
            with open(arabic_file_path, "r", encoding="utf-8") as f:
                surah_arabic_data = json.load(f)
            
            # Asumsi surah_arabic_data adalah objek yang punya kunci 'verse'
            arabic_verses_map = surah_arabic_data['verse'] 

            # Iterasi melalui setiap ayat dalam surah ini berdasarkan jumlah ayat (count)
            for i in range(1, surah_ayat_count + 1): 
                ayat_key = f"verse_{i}"
                if ayat_key in arabic_verses_map:
                    arabic_text = arabic_verses_map[ayat_key]
                else:
                    st.warning(f"Warning: Arabic text for {ayat_key} in Surah {surah_number_str} not found. Skipping this verse.")
                    continue 

                # Membangun path audio lokal (sesuai struktur folder Anda: source/audio/001/1.mp3)
                audio_path = os.path.join(base_path, "audio", surah_number_str, f"{str(i).zfill(3)}.mp3") #
                
                # Tambahkan data ayat ke list utama
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
    base_path = "quran/source" #
    juz_file_path = os.path.join(base_path, "juz.json") #
    
    juz_data_map = {}
    try:
        with open(juz_file_path, "r", encoding="utf-8") as f: #
            juz_list = json.load(f) #
            for juz_info in juz_list: #
                juz_index = int(juz_info['index']) #
                start_info = juz_info['start'] #
                end_info = juz_info['end'] #

                start_surah = int(start_info['index']) #
                start_ayat = int(start_info['verse'].replace('verse_', '')) #
                end_surah = int(end_info['index']) #
                end_ayat = int(end_info['verse'].replace('verse_', '')) #
                
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

# Cek jika data Quran berhasil dimuat
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

    s_surah = juz_range['start_surah']
    s_ayat = juz_range['start_ayat']
    e_surah = juz_range['end_surah']
    e_ayat = juz_range['end_ayat']

    ayat_surah = ayat_data['surah_number']
    ayat_ayat = ayat_data['ayat_number']

    if ayat_surah > s_surah and ayat_surah < e_surah:
        return True
    elif ayat_surah == s_surah and ayat_ayat >= s_ayat:
        return True
    elif ayat_surah == e_surah and ayat_ayat <= e_ayat:
        return True
    return False

@st.cache_data
def filter_quran_by_juz(full_quran_data_list, selected_juz_options, juz_ranges_map_param):
    """Filters the full Quran data list based on selected Juz options."""
    if "30 Juz" in selected_juz_options or not selected_juz_options:
        return full_quran_data_list

    filtered_data = []
    selected_juz_numbers = []
    for opt in selected_juz_options:
        if opt.startswith("Juz "):
            try:
                selected_juz_numbers.append(int(opt.split(" ")[1]))
            except ValueError:
                pass 

    if not selected_juz_numbers:
        return full_quran_data_list

    for ayat_data in full_quran_data_list:
        for juz_num in selected_juz_numbers:
            if is_ayat_in_juz(ayat_data, juz_num, juz_ranges_map_param): 
                filtered_data.append(ayat_data)
                break 
    return filtered_data

def get_next_ayat(current_ayat_info, all_quran_data_list): # Parameter diubah namanya untuk kejelasan
    """
    Mencari ayat selanjutnya berdasarkan informasi ayat saat ini dalam daftar data Al-Qur'an yang diberikan.
    Menangani perpindahan ke surah berikutnya jika sudah di akhir surah.
    """
    current_surah_num = current_ayat_info['surah_number']
    current_ayat_num = current_ayat_info['ayat_number']
    
    # Cari indeks ayat saat ini dalam daftar yang diberikan
    current_index_in_list = -1
    for i, ayat_data in enumerate(all_quran_data_list):
        if ayat_data['surah_number'] == current_surah_num and ayat_data['ayat_number'] == current_ayat_num:
            current_index_in_list = i
            break
    
    if current_index_in_list == -1 or current_index_in_list + 1 >= len(all_quran_data_list):
        return None, -1 # Ayat saat ini tidak ditemukan atau sudah di akhir daftar

    next_ayat_candidate = all_quran_data_list[current_index_in_list + 1]

    # Logika tambahan untuk memastikan next_ayat_candidate secara logis "setelah" current_ayat
    # dalam urutan Al-Qur'an (penting jika daftar sudah difilter/tidak berurutan penuh)
    if (next_ayat_candidate['surah_number'] > current_surah_num) or \
       (next_ayat_candidate['surah_number'] == current_surah_num and next_ayat_candidate['ayat_number'] == current_ayat_num + 1):
        return next_ayat_candidate, current_index_in_list + 1
    
    return None, -1 # Kasus di mana ayat berikutnya dalam filtered_quran_data tidak berurutan


def fuzzy_match_arabic_ayat(original_text_ar, transcribed_text_ar, threshold=0.7):
    """
    Membandingkan dua teks Arab menggunakan Levenshtein distance setelah normalisasi.
    """
    normalized_original = normalize_arabic_text(original_text_ar)
    normalized_transcribed = normalize_arabic_text(transcribed_text_ar)
    
    if not normalized_original:
        return False, 0.0

    lev_distance = distance(normalized_original, normalized_transcribed)
    
    similarity_score = (len(normalized_original) - lev_distance) / len(normalized_original)
    
    return similarity_score >= threshold, similarity_score

# --- Inisialisasi State Aplikasi ---
# current_ayat_index akan menyimpan indeks ayat yang sedang ditampilkan
if 'current_ayat_index' not in st.session_state:
    st.session_state.current_ayat_index = 0 # Default ke ayat pertama
# Kunci untuk widget audio_input agar bisa di-reset saat ganti ayat
if 'audio_input_key' not in st.session_state:
    st.session_state.audio_input_key = 0
# Menyimpan transkripsi terakhir agar tetap terlihat setelah rerun
if 'last_transcription' not in st.session_state:
    st.session_state.last_transcription = ""
# Menyimpan hasil pencocokan terakhir (apakah cocok, skor)
if 'last_match_result' not in st.session_state:
    st.session_state.last_match_result = None # Format: (is_match, score)
# Menyimpan pilihan Juz sebelumnya untuk mendeteksi perubahan filter
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

# Pilihan Juz Multiselect
options = st.multiselect( #
    "Select Juz to play",
    ["30 Juz"] + [f"Juz {i}" for i in range(1, 31)], 
    default=st.session_state.previous_juz_options, # Pertahankan pilihan sebelumnya
    key="juz_selector" 
)

# Filter data Al-Qur'an berdasarkan pilihan Juz
filtered_quran_data = filter_quran_by_juz(full_quran_data, options, juz_ranges_map)

# Cek apakah ada ayat dalam data yang difilter
if not filtered_quran_data:
    st.warning("No verses found for the selected Juz options. Please select other Juz.")
    st.stop() # Hentikan eksekusi jika tidak ada data untuk dimainkan

# Reset current_ayat_index jika pilihan Juz berubah
if st.session_state.get('previous_juz_options') != options:
    st.session_state.current_ayat_index = random.randint(0, len(filtered_quran_data) - 1)
    st.session_state.previous_juz_options = options 
    st.session_state.audio_input_key = 0
    st.session_state.last_transcription = ""
    st.session_state.last_match_result = None
    st.rerun() # Penting untuk me-rerun agar perubahan filter diterapkan

# Mengambil data ayat yang sedang aktif dari list filtered_quran_data
current_ayat = filtered_quran_data[st.session_state.current_ayat_index]
# Mencari ayat selanjutnya yang akan menjadi target pengguna untuk disambung
next_ayat, next_ayat_index = get_next_ayat(current_ayat, filtered_quran_data) # Meneruskan filtered_quran_data

# --- Tampilan Ayat ---
st.markdown(f"### Clue, verse: **{current_ayat['ayat_number']}**")
st.markdown(f"<h1 style='text-align: right; font-size: 3em; direction: rtl;'>{current_ayat['text_ar']}</h1>", unsafe_allow_html=True)

# --- Bagian Audio dan Tombol Randomize ---
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Listen 🔊", use_container_width=True):
        if os.path.exists(current_ayat['audio_path']):
            st.audio(current_ayat['audio_path'], format="audio/mp3", start_time=0, autoplay=True)
        else:
            st.warning(f"Audio file not found: {current_ayat['audio_path']}. Cannot play.")

with col2:
    if st.button("Change Verse ➡️", use_container_width=True):
        new_index = random.randint(0, len(filtered_quran_data) - 1) # Gunakan filtered_quran_data
        if len(filtered_quran_data) > 1:
            while new_index == st.session_state.current_ayat_index:
                new_index = random.randint(0, len(filtered_quran_data) - 1)
            
        st.session_state.current_ayat_index = new_index
        st.session_state.audio_input_key += 1 
        st.session_state.last_transcription = "" 
        st.session_state.last_match_result = None 
        st.rerun()

st.markdown("---")
# Tombol untuk memulai rekaman Speech-to-Text untuk menyambung ayat
if st.button("🎙️ Record Recitation"):
    if next_ayat: 
        status_message_placeholder = st.empty()
        try:
            with st.spinner("Recording audio..."):
                status_message_placeholder.info("Microphone ready. Please start speaking...")
                transcribed_text = speech_to_text_from_mic()

            status_message_placeholder.empty() # Hapus pesan "Silakan berbicara..."
            st.session_state.last_transcription = transcribed_text 
            
            is_match, similarity_score = fuzzy_match_arabic_ayat(next_ayat['text_ar'], transcribed_text)
            st.session_state.last_match_result = (is_match, similarity_score) 
            
            st.success("Transcription complete!")
            st.write("**Your recognized recitation text:**")
            st.code(transcribed_text)
            
            if is_match:
                st.success(f"**🥳 Match! You successfully connected the verse!** (Similarity Score: {similarity_score:.2f})")
            else:
                st.error(f"**😔 No match. Please try again.** (Similarity Score: {similarity_score:.2f})")
                st.info("Ensure you are reciting the target verse correctly.")
        except Exception as e:
            status_message_placeholder.error(f"An error occurred during recording: {e}")
    else:
        st.warning("No next verse to connect.")

# # Tampilkan transkripsi dan hasil pencocokan sebelumnya jika ada (setelah rerun)
# if st.session_state.last_transcription:
#     st.write("**Terakhir kali Anda mengucapkan (dari mikrofon):**")
#     st.code(st.session_state.last_transcription)
#     if st.session_state.last_match_result:
#         is_match, similarity_score = st.session_state.last_match_result
#         if is_match:
#             st.success(f"**🥳 Cocok!** (Skor Kemiripan: {similarity_score:.2f})")
#         else:
#             st.error(f"**😔 Belum cocok.** (Skor Kemiripan: {similarity_score:.2f})")


# st.markdown("---")
# # Bagian untuk Unggah Rekaman (Alternatif) - diaktifkan kembali
# st.subheader("Unggah Rekaman Anda (Alternatif):")
# uploaded_file = st.audio_input("Unggah rekaman audio Anda di sini", key=f"audio_input_{st.session_state.audio_input_key}")

# if uploaded_file is not None:
#     if next_ayat is None:
#         st.warning("Tidak ada ayat selanjutnya untuk disambung. Unggah file tidak dapat diproses.")
#     else:
#         temp_audio_file_path = None
#         try:
#             file_extension = uploaded_file.type.split('/')[-1] if uploaded_file.type else 'wav'
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio_file:
#                 temp_audio_file.write(uploaded_file.read())
#                 temp_audio_file_path = temp_audio_file.name

#             st.write("File audio berhasil diunggah!")
#             st.audio(uploaded_file, format=uploaded_file.type) 

#             st.info("Menganalisis audio Anda dari file untuk transkripsi...")

#             speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
#             speech_config.speech_recognition_language = "ar-SA" 
            
#             audio_input_config = speechsdk.audio.AudioConfig(filename=temp_audio_file_path)
#             speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input_config)

#             with st.spinner("Mengonversi audio dari file ke teks..."):
#                 result = speech_recognizer.recognize_once_async().get()

#             transcribed_text_from_file = ""
#             if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#                 transcribed_text_from_file = result.text
#                 st.success("Transkripsi Selesai!")
#             elif result.reason == speechsdk.ResultReason.NoMatch:
#                 transcribed_text_from_file = "Tidak dapat mengenali ucapan dari file."
#                 st.warning(transcribed_text_from_file)
#             elif result.reason == speechsdk.ResultReason.Canceled:
#                 cancellation_details = result.cancellation_details
#                 if cancellation_details.reason == speechsdk.CancellationReason.Error:
#                     transcribed_text_from_file = f"Error: {cancellation_details.error_details}"
#                 else:
#                     transcribed_text_from_file = f"Dibatalkan: {cancellation_details.reason}"
#                 st.error(transcribed_text_from_file)
#             else:
#                 transcribed_text_from_file = "Terjadi kesalahan yang tidak diketahui saat memproses file."
#                 st.error(transcribed_text_from_file)

#             st.write("**Teks Bacaan Anda (dari file) yang Dikenali:**")
#             st.code(transcribed_text_from_file)

#             if transcribed_text_from_file and transcribed_text_from_file != "Tidak dapat mengenali ucapan dari file.":
#                 is_match_file, similarity_score_file = fuzzy_match_arabic_ayat(next_ayat['text_ar'], transcribed_text_from_file)
                
#                 st.markdown("---")
#                 if is_match_file:
#                     st.success(f"**🥳 Cocok! Anda berhasil menyambung ayat dari file!** (Skor Kemiripan: {similarity_score_file:.2f})")
#                 else:
#                     st.error(f"**😔 Belum cocok dari file. Mohon coba lagi.** (Skor Kemiripan: {similarity_score_file:.2f})")
#                     st.info("Pastikan file audio Anda mengandung bacaan ayat target dengan jelas.")
            
#         except Exception as e:
#             st.error(f"Terjadi kesalahan saat memproses file audio: {e}")
#         finally:
#             if temp_audio_file_path and os.path.exists(temp_audio_file_path):
#                 os.remove(temp_audio_file_path)

st.markdown("---")
with st.expander("Answer Key", icon="💡", expanded=False):
    if next_ayat:
        st.markdown(f"### Surah: **{next_ayat['surah_name_ar']}**, Verse: **{next_ayat['ayat_number']}**")
        st.markdown(f"<h3 style='text-align: right; font-size: 2em; direction: rtl;'>{next_ayat['text_ar']}</h3>", unsafe_allow_html=True)
    else:
        st.warning("No target verse available. You have reached the end of the Quran.")