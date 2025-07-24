import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

def home():
    st.title("Welcome 👋🏻")
    header = st.container()
    header.subheader("Let's chit chat with Wafiq Assistant!")

    if 'pills_state' not in st.session_state:
        st.session_state.pills_state = [
            {"label": "Hello, how are you?", "clicked": False},
            {"label": "Who is Wafiq?", "clicked": False},
            {"label": "Where he was born?", "clicked": False},
            {"label": "How old is he?", "clicked": False}
        ]

    current_options_labels = [item["label"] for item in st.session_state.pills_state]
    selection = header.pills("", current_options_labels)

    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    ### Custom CSS for the sticky header
    st.markdown(
        """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            background-color: white;
            z-index: 999;
        }
        .fixed-header {
            border-bottom: 3px solid grey;
        }
    </style>
        """,
        unsafe_allow_html=True
    )

    try:
        genai.configure(api_key=os.environ['NAMADAUN'])
        # Pilih model Gemini. 'gemini-pro' cocok untuk teks.
        # Ada juga 'gemini-pro-vision' untuk multimodal.
        model = genai.GenerativeModel('gemini-2.5-flash')
    except KeyError:
        st.error("Kunci API Google Gemini tidak ditemukan! Pastikan Anda menyimpannya di `.streamlit/secrets.toml`.")
        st.stop() # Hentikan eksekusi jika kunci API tidak ada

    # --- Inisialisasi Riwayat Obrolan ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    opening = st.chat_message("assistant")
    opening.write("Hi, I'm Wafiq Assistant. Any things that you want to ask?")

    # --- Tampilkan Riwayat Obrolan Sebelumnya ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    def send_message(user_message):
        # Tambahkan pesan pengguna ke riwayat
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

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
                    response = chat.send_message(user_message) # Kirim prompt user saat ini
                    
                    # Tampilkan respons
                    st.markdown(response.text)
                    # Tambahkan respons AI ke riwayat
                    st.session_state.messages.append({"role": "assistant", "content": response.text})

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses permintaan: {e}")
                    st.markdown("Coba lagi atau periksa koneksi internet Anda.")
    
    # --- Logika untuk Pills yang Diklik ---
    if selection:        
        # Kirim pilihan pills sebagai pesan pengguna
        for item in st.session_state.pills_state:
            if item["label"] == selection and not item["clicked"]:
                item["clicked"] = True
                send_message(selection)
                st.rerun()

    # --- Input Pengguna ---
    if prompt := st.chat_input("Ketik pesan Anda di sini..."):
        send_message(prompt)

pg = st.navigation([
    st.Page(home, title="Home", icon="🏠"),
    st.Page("about.py", title="About", icon="🧒🏻"),
    st.Page("project.py", title="Project", icon="💻"),
    st.Page("resume.py", title="Resume", icon="📄"),
    st.Page("contact.py", title="Contact", icon="📲")])

pg.run()