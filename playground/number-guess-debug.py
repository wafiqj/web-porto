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
stroke_width = st.slider("✏️ Brush size:", 5, 50, 20)
stroke_color = st.color_picker("🎨 Choose brush color", "#000000")  # default hitam
bg_color = st.color_picker("🧻 Choose background color", "#FFFFFF")  # default putih

# --- Kanvas gambar ---
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # transparan (biar gak nutup gambar)
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

# Inisialisasi session_state untuk hasil prediksi
if "last_features" not in st.session_state:
    st.session_state.last_features = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_input_array" not in st.session_state:
    st.session_state.last_input_array = None
if "input_tensor" not in st.session_state:
    st.session_state.input_tensor = None

# === Tombol Prediksi ===
if st.button("Prediksi Angka", type="primary"):
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data
        pil_img = Image.fromarray(img_data.astype('uint8'), 'RGBA')

        # --- Preprocessing ---
        rgb_img = pil_img.convert("RGB")
        yolo_input_size = 224
        resized_img = rgb_img.resize((yolo_input_size, yolo_input_size), Image.LANCZOS)
        input_array = np.array(resized_img)
        input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # --- Simpan input ---
        st.session_state.last_input_array = input_array
        st.session_state.input_tensor = input_tensor

        # --- Jalankan model dan ambil fitur ---
        torch_model = model.model
        layer_index = 2  # default layer tengah

        features = {}
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        torch_model.model[layer_index].register_forward_hook(get_features(f"layer_{layer_index}"))
        _ = torch_model(input_tensor)

        # --- Simpan hasil feature map & prediksi ---
        st.session_state.last_features = features
        results = model.predict(source=input_array, verbose=False)
        st.session_state.last_results = results

        st.success("✅ Prediksi berhasil! Sekarang kamu bisa eksplor layer dan channel di bawah.")
    else:
        st.warning("Silakan gambar angka di kanvas terlebih dahulu!")

# === BAGIAN HASIL (selalu tampil jika sudah ada hasil sebelumnya) ===
if st.session_state.last_results is not None and st.session_state.last_features is not None:
    results = st.session_state.last_results
    input_array = st.session_state.last_input_array
    input_tensor = st.session_state.input_tensor
    torch_model = model.model

    st.markdown("## 🔍 Hasil Prediksi & Visualisasi Feature Map")

    # --- Pilih layer ---
    layer_index = st.number_input(
        "Pilih index layer untuk ditampilkan (0 - {})".format(len(torch_model.model) - 1),
        min_value=0,
        max_value=len(torch_model.model) - 1,
        value=2,
        step=1,
        key="layer_index"
    )

    # --- Jika layer berbeda, ambil ulang feature map ---
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    torch_model.model[int(layer_index)].register_forward_hook(get_features(f"layer_{layer_index}"))
    _ = torch_model(input_tensor)
    st.session_state.last_features = features

    feat = features[f"layer_{layer_index}"]
    st.write(f"✅ Feature map shape dari layer {layer_index}: {feat.shape}")

    # --- Pilih channel untuk divisualisasikan ---
    num_channels = feat.shape[1]
    channel_to_show = st.slider(
        "Pilih channel untuk ditampilkan", 
        0, num_channels - 1, 0, key="channel_slider"
    )

    feature_img = feat[0, channel_to_show].cpu().numpy()
    st.image(feature_img, caption=f"Feature Map - Layer {layer_index} (Channel {channel_to_show})", 
             use_column_width=True, clamp=True)

    # --- Tampilkan hasil klasifikasi ---
    if results and hasattr(results[0], 'probs'):
        probs = results[0].probs
        predicted_class_id = probs.top1
        confidence = probs.top1conf.item() * 100
        predicted_label = model.names[predicted_class_id]

        st.markdown(f"### 🤖 AI Prediction: **{predicted_label}** ({confidence:.2f}% confidence)")

        prob_df = pd.DataFrame({
            'Digit': [model.names[i] for i in range(len(model.names))],
            'Probability': probs.data.cpu().numpy()
        })
        st.bar_chart(prob_df.set_index('Digit'))
    else:
        st.error("Model tidak menghasilkan probabilitas klasifikasi yang diharapkan.")
