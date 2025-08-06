import streamlit as st
import re
import nltk

st.markdown("# 🧐 Mood Detector")
st.markdown("Tags : :blue-badge[Sentiment Analysis], :blue-badge[Word Embedding], :blue-badge[Data Mining]")
st.info("This playground is coming soon. Stay tuned for updates!")

# positive_words_id = ["bagus", "hebat", "suka", "menyenangkan", "cinta", "baik", "fantastis", "luar biasa", "oke", "mantap"]
# negative_words_id = ["buruk", "jelek", "kecewa", "tidak suka", "menjijikkan", "marah", "sedih", "parah", "aneh", "rusak"]

# def analyze_sentiment_simple(text):
#     # Pra-pemrosesan: tokenisasi, lowercase, hapus non-alfabet
#     words = re.findall(r'\b[a-z]+\b', text.lower())
    
#     pos_count = 0
#     neg_count = 0
    
#     matched_pos_words = []
#     matched_neg_words = []

#     for word in words:
#         if word in positive_words_id:
#             pos_count += 1
#             matched_pos_words.append(word)
#         elif word in negative_words_id:
#             neg_count += 1
#             matched_neg_words.append(word)
            
#     if pos_count > neg_count:
#         sentiment = "Positif"
#         score = pos_count - neg_count
#     elif neg_count > pos_count:
#         sentiment = "Negatif"
#         score = neg_count - pos_count # Negative score magnitude
#     else:
#         sentiment = "Netral"
#         score = 0
        
#     return sentiment, score, words, matched_pos_words, matched_neg_words

# # st.title("💡 Analisis Sentimen Interaktif")
# st.markdown("Coba masukkan teks di bawah ini dan lihat bagaimana AI menganalisis 'perasaan' di dalamnya!")

# # --- Input Pengguna ---
# user_text = st.text_area("Masukkan teks Anda di sini:", 
#                          "Saya sangat suka produk ini, kualitasnya bagus sekali dan pelayanannya luar biasa.",
#                          height=150)

# if st.button("Analisis Sentimen", type="primary"):
#     if user_text:
#         sentiment, score, cleaned_words, matched_pos, matched_neg = analyze_sentiment_simple(user_text)
        
#         st.markdown("---")
#         st.subheader("📊 Hasil Analisis Sentimen:")
        
#         if sentiment == "Positif":
#             st.success(f"Sentimen: **{sentiment}** 😊")
#         elif sentiment == "Negatif":
#             st.error(f"Sentimen: **{sentiment}** 😠")
#         else:
#             st.info(f"Sentimen: **{sentiment}** 😐")

#         st.progress(max(0.1, abs(score)) / max(0.1, max(len(positive_words_id), len(negative_words_id), abs(score))), text=f"Skor Relatif: {score}")

#         st.markdown("---")
#         st.subheader("🔍 Alur Proses Analisis Sentimen:")

#         # --- Alur Proses ---
#         with st.expander("Langkah 1: Teks Asli"):
#             st.markdown("Ini adalah teks 'mentah' yang Anda masukkan. Sistem akan mulai memprosesnya.")
#             st.code(user_text, language='text')

#         with st.expander("Langkah 2: Membersihkan & Memecah Kata (Pra-pemrosesan)"):
#             st.markdown("Komputer akan memecah teks menjadi kata-kata (token) dan mengubahnya menjadi huruf kecil agar lebih mudah diproses. Angka dan tanda baca biasanya dibuang.")
#             st.write("Kata-kata yang terdeteksi setelah dibersihkan:")
#             st.code(str(cleaned_words), language='python')
#             st.write("*(Misalnya, 'Selamat pagi!' menjadi ['selamat', 'pagi'])*")

#         with st.expander("Langkah 3: Mencari Kata dengan 'Rasa' (Identifikasi Kata Kunci)"):
#             st.markdown("Sistem kemudian mencari kata-kata yang sudah dikenal memiliki 'rasa' positif atau negatif dalam daftar kamus internalnya.")
#             col_pos, col_neg = st.columns(2)
#             with col_pos:
#                 st.markdown("**Kata Positif Ditemukan:**")
#                 if matched_pos:
#                     for word in matched_pos:
#                         st.markdown(f"- **{word}** ✅")
#                 else:
#                     st.markdown("Tidak ada kata positif yang cocok.")
#             with col_neg:
#                 st.markdown("**Kata Negatif Ditemukan:**")
#                 if matched_neg:
#                     for word in matched_neg:
#                         st.markdown(f"- **{word}** ❌")
#                 else:
#                     st.markdown("Tidak ada kata negatif yang cocok.")
            
#             st.info("Kamus kata positif dan negatif yang digunakan dalam demo ini sederhana. Model AI asli menggunakan kamus yang jauh lebih besar dan kompleks!")

#         with st.expander("Langkah 4: Menghitung 'Skor' Perasaan (Penilaian Sentimen)"):
#             st.markdown("Setelah menemukan kata-kata 'berasa', sistem menghitung skor berdasarkan jumlah kemunculan kata positif dan negatif.")
#             st.write(f"- Jumlah Kata Positif: **{len(matched_pos)}**")
#             st.write(f"- Jumlah Kata Negatif: **{len(matched_neg)}**")
#             st.write(f"**Skor Perasaan Awal:** {len(matched_pos)} (positif) - {len(matched_neg)} (negatif) = **{score}**")
#             st.info("Skor ini kemudian digunakan untuk menentukan sentimen akhir.")
        
#         with st.expander("Langkah 5: Kesimpulan Sentimen"):
#             st.markdown("Berdasarkan skor yang dihitung, sistem menyimpulkan 'perasaan' umum dari teks Anda.")
#             st.markdown(f"**Kesimpulan Akhir:** Teks Anda memiliki sentimen **{sentiment}**.")
            
#     else:
#         st.warning("Silakan masukkan teks untuk dianalisis.")

# st.markdown("---")
# st.markdown("### 💡 Bagaimana ini bisa dikembangkan lebih lanjut:")
# st.markdown("""
# * **Akurasi Lebih Tinggi:** Menggunakan model AI terlatih (seperti BERT, Transformer) dari pustaka seperti `Hugging Face Transformers` atau API cloud (Google, OpenAI) yang jauh lebih akurat dan memahami konteks.
# * **Deteksi Emosi Spesifik:** Mendeteksi emosi yang lebih detail seperti kebahagiaan, kemarahan, kesedihan, dll.
# * **Visualisasi Lanjutan:** Menampilkan *word cloud* atau grafik distribusi sentimen.
# """)