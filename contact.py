import streamlit as st
import google.generativeai as genai

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Chatbot Demo",
    page_icon="🤖",
    layout="centered"
)

st.title("Chatbot Demo")

# --- Inisialisasi Model Gemini ---
# Pastikan kunci API Anda ada di .streamlit/secrets.toml
try:
    genai.configure(api_key="AIzaSyBGfprRY9PW9PED_C_XcqPAZwHatxfMkbo")
    # Pilih model Gemini. 'gemini-pro' cocok untuk teks.
    # Ada juga 'gemini-pro-vision' untuk multimodal.
    model = genai.GenerativeModel('gemini-2.5-flash')
except KeyError:
    st.error("Kunci API Google Gemini tidak ditemukan! Pastikan Anda menyimpannya di `.streamlit/secrets.toml`.")
    st.stop() # Hentikan eksekusi jika kunci API tidak ada

# st.markdown(
#     """
# <style>
#     .st-emotion-cache-1mph9ef {
#         flex-direction: row-reverse;
#         text-align: right;
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )


@st.dialog("AI-Chat Demo")
def chat():

    # --- Inisialisasi Riwayat Obrolan ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Tampilkan Riwayat Obrolan Sebelumnya ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Input Pengguna ---
    if prompt := st.chat_input("Ketik pesan Anda di sini..."):
        # Tambahkan pesan pengguna ke riwayat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Dapatkan respons dari model Gemini
        with st.chat_message("assistant"):
            with st.spinner("Memikirkan jawaban..."):
                try:
                    # Kirim riwayat obrolan untuk konteks
                    # Konversi format riwayat untuk model Gemini
                    history_for_gemini = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            history_for_gemini.append({"role": "user", "parts": [msg["content"]]})
                        elif msg["role"] == "assistant":
                            history_for_gemini.append({"role": "model", "parts": [msg["content"]]})
                    
                    # Mulai chat dengan riwayat yang ada
                    chat = model.start_chat(history=history_for_gemini[:-1]) # Kirim semua kecuali pesan terakhir (prompt user saat ini)
                    response = chat.send_message(prompt) # Kirim prompt user saat ini
                    
                    # Tampilkan respons
                    st.markdown(response.text)
                    # Tambahkan respons AI ke riwayat
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses permintaan: {e}")
                    st.markdown("Coba lagi atau periksa koneksi internet Anda.")


if "vote" not in st.session_state:
    if st.button("Chat now"):
        chat()
