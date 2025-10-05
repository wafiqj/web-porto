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
if st.button("Prediksi Angka", type="primary"):
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data
        pil_img = Image.fromarray(img_data.astype('uint8'), 'RGBA')

        # Konversi ke RGB dan resize
        rgb_img = pil_img.convert("RGB")
        yolo_input_size = 224
        resized_img = rgb_img.resize((yolo_input_size, yolo_input_size), Image.LANCZOS)
        input_array = np.array(resized_img)

        # Prediksi
        results = model.predict(source=input_array, verbose=False)

        if results and hasattr(results[0], 'probs'):
            probs = results[0].probs
            predicted_class_id = probs.top1
            confidence = probs.top1conf.item() * 100
            predicted_label = model.names[predicted_class_id]

            st.markdown(f"### AI Memprediksi Angka: **{predicted_label}**")
            st.markdown(f"Dengan Keyakinan: **{confidence:.2f}%**")

            prob_df = pd.DataFrame({
                'Angka': [model.names[i] for i in range(len(model.names))],
                'Probabilitas': probs.data.cpu().numpy()
            })
            st.bar_chart(prob_df.set_index('Angka'))

            # ---- Langkah 4: Heatmap Input ----
            with st.expander("Langkah 4: Heatmap Input"):
                fig, ax = plt.subplots()
                ax.imshow(resized_img, cmap="gray")
                ax.set_title("Heatmap Input ke Model")
                st.pyplot(fig)

            # ---- Langkah 5: Feature Map Explorer ----
            with st.expander("Langkah 5: Feature Map Explorer"):
                st.write("Geser slider untuk melihat feature map dari layer berbeda.")
                tensor_input = torch.from_numpy(input_array).permute(2,0,1).unsqueeze(0).float()/255.0

                layer_idx = st.slider("Pilih layer untuk visualisasi", 0, len(model.model)-1, 0)
                features = model.model[layer_idx](tensor_input)

                # tampilkan 8 feature map pertama
                num_features = min(8, features.shape[1])
                fig, axes = plt.subplots(1, num_features, figsize=(15,3))
                for i in range(num_features):
                    axes[i].imshow(features[0,i].detach().cpu().numpy(), cmap="viridis")
                    axes[i].axis("off")
                st.pyplot(fig)

            # ---- Extra: Grad-CAM ----
            with st.expander("Visualisasi Grad-CAM"):
                st.write("Menunjukkan area pada gambar yang paling berpengaruh pada prediksi.")

                # Grad-CAM sederhana di conv terakhir
                target_layer = model.model[-2]
                grad_cam_input = tensor_input.clone().requires_grad_(True)
                act = {}
                grad = {}

                def forward_hook(module, inp, out):
                    act['value'] = out
                def backward_hook(module, ginp, gout):
                    grad['value'] = gout[0]

                target_layer.register_forward_hook(forward_hook)
                target_layer.register_backward_hook(backward_hook)

                out = model.model(grad_cam_input)
                class_score = out[0, predicted_class_id]
                class_score.backward()

                weights = grad['value'][0].mean(dim=(1,2))
                cam = torch.zeros(act['value'].shape[2:])
                for i, w in enumerate(weights):
                    cam += w * act['value'][0,i]
                cam = torch.relu(cam)
                cam = cam / cam.max()

                cam_resized = cv2.resize(cam.detach().cpu().numpy(), (yolo_input_size, yolo_input_size))
                heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(np.array(resized_img), 0.5, heatmap, 0.5, 0)

                st.image(overlay, caption="Grad-CAM Overlay")
        else:
            st.error("Model tidak menghasilkan probabilitas klasifikasi yang diharapkan.")
    else:
        st.warning("Silakan gambar angka di kanvas terlebih dahulu!")
