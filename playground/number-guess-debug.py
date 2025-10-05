import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
import cv2

st.markdown("# 🔢 Guess the Number")
st.markdown("Tags : :blue-badge[Image Classification], :blue-badge[Deep Learning], :blue-badge[Computer Vision]")

@st.cache_resource
def load_yolo_classify_model():
    model_path = "assets/best.pt"
    model = YOLO(model_path)
    return model

model = load_yolo_classify_model()

st.markdown("Mari kita lihat bagaimana AI 'melihat' dan memprediksi angka yang Anda gambar menggunakan model YOLO!")

# --- Pengaturan warna & ukuran kanvas ---
canvas_width = 280
canvas_height = 280

left_col, right_col = st.columns(["1, 2, "auto"])

with right_col:
    # st.subheader("🖌️ Settings")
    stroke_width = st.slider("Brush size", 1, 50, 25)
    stroke_color = st.color_picker("Brush color", "#FFFFFF")
    bg_color = st.color_picker("Background", "#000000")

with left_col:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key="canvas_drawing",
    )

# --- Tombol Prediksi ---
st.markdown("---")
if st.button("Prediksi Angka", type="primary"):
    if canvas_result.image_data is not None:
        # Konversi data gambar dari kanvas ke PIL Image
        img_data = canvas_result.image_data # RGBA numpy array
        pil_img = Image.fromarray(img_data.astype('uint8'), 'RGBA')

        st.subheader("🔍 Alur Proses Prediksi AI (YOLO):")

        # --- Langkah 2: Gambar Asli Anda ---
        with st.expander("Langkah 2: Gambar Asli yang Anda Gambar"):
            st.markdown("Ini adalah gambar angka yang baru saja Anda gambar di kanvas.")
            st.image(pil_img, caption="Gambar Asli", use_column_width=False, width=280)

        # --- Langkah 3: Pra-pemrosesan Gambar (Agar AI Mengerti) ---
        with st.expander("Langkah 3: Pra-pemrosesan Gambar (Sesuai Kebutuhan YOLO!)"):
            st.markdown("Model AI (YOLO) memiliki format input yang spesifik. Gambar Anda akan diubah agar sesuai:")
            
            # Konversi RGBA ke RGB (YOLO classification butuh RGB)
            st.markdown("#### 3.1 Konversi ke RGB")
            st.markdown("Gambar dari kanvas (RGBA) diubah menjadi format RGB (Red, Green, Blue) yang umum digunakan oleh model Deep Learning.")
            rgb_img = pil_img.convert("RGB")
            st.image(rgb_img, caption="Setelah Konversi ke RGB", use_column_width=False, width=280)
            
            # Ukuran input umum untuk YOLO classification adalah 224x224, 640x640, dll.
            # Kita akan resize ke 224x224 sebagai contoh umum.
            yolo_input_size = 224 
            st.markdown(f"#### 3.2 Ubah Ukuran ke {yolo_input_size}x{yolo_input_size} Piksel")
            st.markdown(f"Model YOLO dilatih dengan gambar berukuran {yolo_input_size}x{yolo_input_size} piksel. Gambar Anda akan diubah ukurannya agar sesuai.")
            resized_img = rgb_img.resize((yolo_input_size, yolo_input_size), Image.LANCZOS)
            st.image(resized_img, caption=f"Setelah Ukuran {yolo_input_size}x{yolo_input_size}", use_column_width=False, width=280)
            
            # Konversi ke NumPy array (model YOLO menerima numpy array atau PIL Image)
            input_array = np.array(resized_img)

            # 🔹 Konversi ke tensor PyTorch
            input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            st.write("Bentuk data akhir yang masuk ke model: `(tinggi, lebar, channel)`")
            st.code(str(input_array.shape), language='python') 
            st.caption("Ini berarti gambar dengan tinggi dan lebar tertentu, serta 3 channel warna (RGB). Normalisasi piksel (0-255 menjadi 0-1) biasanya diurus secara internal oleh model YOLO.")

        # --- Langkah 4: Model Menerima Input (Lapisan Input) ---
        with st.expander("Langkah 4: Model YOLO 'Melihat' Angka Anda (Lapisan Input)"):
            st.markdown("Gambar yang sudah diproses ini (RGB, ukuran tertentu) kemudian dimasukkan ke lapisan input model YOLO.")
            st.markdown("Model ini didesain untuk memproses gambar secara efisien dan mulai mengekstrak informasi.")
            
            # # Ambil model PyTorch di dalam YOLO
            # torch_model = model.model

            # # Pilih layer yang ingin di-inspect
            # layer_index = st.number_input("🔢 Pilih index layer yang ingin dilihat (0 - {}):".format(len(torch_model.model)-1), 
            #                             min_value=0, 
            #                             max_value=len(torch_model.model)-1, 
            #                             value=2, 
            #                             step=1)
            
            # features = {}
            # def get_features(name):
            #     def hook(model, input, output):
            #         features[name] = output.detach()
            #     return hook

            # # # Pasang hook di layer yang dipilih
            # handle = torch_model.model[layer_index].register_forward_hook(get_features(f"layer_{layer_index}"))
            # _ = torch_model(input_tensor)
            # handle.remove()  # cegah hook aktif di model.predict()
            # # Jalankan forward pass manual
            # _ = torch_model(input_tensor)

            # # Ambil hasil fitur
            # feat = features[f"layer_{layer_index}"]
            # st.write(f"✅ Feature map shape dari layer {layer_index}:", feat.shape)

            # # 🔹 Visualisasi salah satu channel dari feature map
            # num_channels = feat.shape[1]
            # channel_to_show = st.slider("Pilih channel untuk ditampilkan", 0, num_channels-1, 0)
            # feature_img = feat[0, channel_to_show].cpu().numpy()

            # st.image(feature_img, caption=f"Feature Map - Layer {layer_index} (Channel {channel_to_show})", 
            #         use_column_width=True, clamp=True)

        # --- Langkah 5: Jaringan Saraf Belajar Fitur (Lapisan Tersembunyi) ---
        with st.expander("Langkah 5: Jaringan Saraf 'Belajar' Pola (Lapisan Tersembunyi)"):
            st.markdown("Di dalam model YOLO, ada banyak 'lapisan tersembunyi' yang bekerja seperti detektif visual.")
            st.markdown("Setiap lapisan secara otomatis belajar mendeteksi fitur atau pola yang semakin kompleks pada gambar:")
            st.markdown("- Lapisan awal mungkin fokus pada **garis, sudut, atau tekstur dasar**.")
            st.markdown("- Lapisan selanjutnya menggabungkan fitur-fitur dasar ini untuk mengenali **bentuk-bentuk angka** secara keseluruhan.")
            st.markdown("YOLO secara khusus sangat efisien dalam mengenali pola-pola ini dengan cepat.")

        # --- Langkah 6: Prediksi & Probabilitas (Lapisan Output) ---
        with st.expander("Langkah 6: AI 'Memutuskan' dan Memberi Keyakinan (Lapisan Output)"):
            st.markdown("Setelah memproses semua pola, lapisan terakhir model YOLO akan mengeluarkan 'suara' atau **probabilitas** untuk setiap kelas angka (0-9).")
            st.markdown("Probabilitas ini menunjukkan seberapa yakin AI bahwa gambar yang Anda berikan adalah angka tertentu.")
            
            # Lakukan prediksi menggunakan model YOLO
            # 'predict' method returns a list of Results objects
            results = model.predict(source=input_array, verbose=False) 
            
            # Asumsi: model classification akan memiliki atribut 'probs' pada objek Results
            # dan 'names' untuk nama-nama kelas.
            if results and hasattr(results[0], 'probs'):
                probs = results[0].probs # Probabilities object
                predicted_class_id = probs.top1 # Index of the top 1 prediction
                confidence = probs.top1conf.item() * 100 # Confidence score
                
                # Mendapatkan nama kelas dari model (misalnya '0', '1', ..., '9')
                # Asumsi model.names berisi ['0', '1', ..., '9']
                predicted_label = model.names[predicted_class_id]
                
                st.markdown(f"### AI Memprediksi Angka: **{predicted_label}**")
                st.markdown(f"Dengan Keyakinan: **{confidence:.2f}%**")

                st.markdown("#### Probabilitas untuk Setiap Angka:")
                # Tampilkan probabilitas dalam bentuk bar chart
                prob_df = pd.DataFrame({
                    'Angka': [model.names[i] for i in range(len(model.names))],
                    'Probabilitas': probs.data.cpu().numpy() # Pastikan data numpy
                })
                st.bar_chart(prob_df.set_index('Angka'))
                st.caption("Batang tertinggi menunjukkan angka yang paling mungkin menurut AI.")
            else:
                st.error("Model tidak menghasilkan probabilitas klasifikasi yang diharapkan. Pastikan ini adalah model klasifikasi YOLO yang benar.")


    else:
        st.warning("Silakan gambar angka di kanvas terlebih dahulu!")

st.markdown("---")
st.markdown("### 💡 Apa yang Anda Pelajari:")
st.markdown("""
* **Pra-pemrosesan Khusus:** Setiap model AI membutuhkan input dengan format (ukuran, channel warna) yang spesifik.
* **YOLO untuk Klasifikasi:** YOLO tidak hanya untuk deteksi objek, tapi juga sangat efisien dalam tugas klasifikasi gambar.
* **Keyakinan AI:** AI tidak 100% yakin, tapi memberikan 'keyakinan' dalam bentuk probabilitas untuk setiap kemungkinan.
""")
st.markdown("Coba gambar angka yang berbeda dan lihat bagaimana AI memprediksinya!")