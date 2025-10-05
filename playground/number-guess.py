import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
# import cv2

# --- Title and Description ---
st.markdown("# 🔢 Guess the Number :blue-badge[by AI, not you.]")
st.markdown("Tags : :blue-badge[Image Classification], :blue-badge[Deep Learning], :blue-badge[Computer Vision]")
st.markdown("Tech : :red-badge[Streamlit Canvas], :red-badge[YOLO], :red-badge[Pandas], :red-badge[Numpy], :red-badge[PyTorch], :red-badge[Pillow], :red-badge[Matplotlib]")

@st.cache_resource
def load_yolo_classify_model():
    model_path = "assets/best.pt"
    model = YOLO(model_path)
    return model

model = load_yolo_classify_model()

with st.expander("📝 Readme", expanded=True):
    st.markdown("""
    **Guess the Number** is an interactive playground that showcases how AI recognizes handwritten digits.  
    It uses Deep Learning and Computer Vision to analyze your drawing and predict which number you wrote.

    **How to Play:**
    1. Draw any digit (0–9) on the canvas.
    2. **(optional)** Adjust brush color, size, or background as you like.
    3. Click **Guess the number** to let the AI predict.
    4. Explore how the model processes your drawing — from preprocessing, feature extraction, to final prediction.

    **Features:**
    - Interactive drawing canvas with customizable brush and background.
    - Step-by-step AI visualization (preprocessing → feature learning → classification).
    - Confidence chart showing model probabilities.

    Draw any digit you like, and let AI guess your number. Whether it’s a perfect ‘8’ or a messy ‘2’, the model will do its best to recognize it — just like a mini handwriting recognition powered by deep learning. ✨
    """)
st.markdown("---")

# --- Canvas Settings ---
canvas_width = 340
canvas_height = 280

one_col, two_col, three_col = st.columns([1,1,1], gap="medium")

with two_col:
    stroke_width = st.slider("Brush size", 1, 50, 25)
    stroke_color = st.color_picker("Brush color", "#FFFFFF")
    # bg_color = st.color_picker("Background", "#000000")

with one_col:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#000000",
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key="canvas_drawing",
    )

# --- Prediction Button ---
if st.button("Guess the number", type="primary"):
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data
        pil_img = Image.fromarray(img_data.astype('uint8'), 'RGBA')

        st.markdown("---")
        st.subheader("🧠 How AI Understands Your Drawing")

        # ============================================================
        # 1️⃣ Step 1 — Preprocessing
        # ============================================================
        with st.expander("Step 1: Preprocessing the Image"):
            st.markdown("""
            Your drawing is first converted into a format that the model can understand —  
            resized, cleaned, and normalized.
            """)

            # Convert RGBA → RGB
            rgb_img = pil_img.convert("RGB")

            # Resize for YOLO input
            yolo_input_size = 224
            resized_img = rgb_img.resize((yolo_input_size, yolo_input_size), Image.LANCZOS)

            # Convert to numpy and tensor
            input_array = np.array(resized_img)
            input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # --- Visual layout: original → preprocessed ---
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.image(pil_img, caption=f"🖋️ Original Drawing ({canvas_width}x{canvas_height})")
            with col2:
                st.image(resized_img, caption=f"✨ Preprocessed ({yolo_input_size}×{yolo_input_size})")

            st.markdown(" ")
            st.code(f"Final input shape: {input_array.shape}", language='python')


        # ============================================================
        # 2️⃣ Step 2 — Model Learns Visual Patterns
        # ============================================================
        with st.expander("Step 2: The Model Learns Visual Patterns"):
            st.markdown("""
            Inside YOLO, multiple convolutional layers process your image step by step —  
            starting from simple edges and strokes, then evolving into complete digit shapes.  
            Below are the visual representations (*feature maps*) from several key layers.
            """)

            important_layers = [0, 1, 2, 3]  # adjust if needed
            features = {}

            torch_model = model.model.model  # YOLO backbone

            def get_features(name):
                def hook(m, i, o):
                    features[name] = o.detach().cpu()
                return hook

            handles = []
            for i in important_layers:
                h = torch_model[i].register_forward_hook(get_features(f"Layer {i}"))
                handles.append(h)

            # single forward pass
            _ = torch_model(input_tensor)

            # remove hooks
            for h in handles:
                h.remove()

            # Prepare 2x2 grid: for each important layer, build one horizontal combined image (up to 4 channels)
            combined_images = []
            layer_infos = []
            for name in [f"Layer {i}" for i in important_layers]:
                feat = features.get(name)  # tensor shape: (1, C, H, W)
                if feat is None:
                    # fallback blank image
                    blank = np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.uint8)
                    combined_images.append(blank)
                    layer_infos.append((name, None))
                    continue

                feat_np = feat.numpy()  # (1, C, H, W)
                C = feat_np.shape[1]
                num_to_show = min(4, C)

                # build channel images (normalize each channel individually)
                channel_imgs = []
                for j in range(num_to_show):
                    arr = feat_np[0, j]
                    arr = arr - arr.min()
                    maxv = arr.max()
                    if maxv > 0:
                        arr = arr / maxv
                    arr = (arr * 255).astype(np.uint8)
                    channel_imgs.append(arr)

                # if less than 4 channels, pad with empty images
                h, w = channel_imgs[0].shape if channel_imgs else (input_tensor.shape[2], input_tensor.shape[3])
                while len(channel_imgs) < 4:
                    channel_imgs.append(np.zeros((h, w), dtype=np.uint8))

                # concat horizontally
                combined = np.concatenate(channel_imgs, axis=1)  # shape (h, 4*w)
                combined_images.append(combined)
                layer_infos.append((name, tuple(feat.shape)))

            # Now plot the 2x2 grid where each cell is the corresponding 'combined' image
            fig, axes = plt.subplots(2, 2, figsize=(10, 4))  # square layout
            axes = axes.flatten()

            for idx, ax in enumerate(axes):
                if idx < len(combined_images):
                    img = combined_images[idx]

                    # --- ✅ Preserve square shape ---
                    ax.imshow(img, cmap='gray', interpolation='nearest')
                    ax.set_aspect('equal')  # ensure square aspect ratio
                    ax.axis('off')

                    # --- Caption under each cell ---
                    name, shape = layer_infos[idx]
                    caption = f"{name}"
                    if shape is not None:
                        caption += f" · shape: {shape}"
                    ax.set_title(caption, fontsize=9, pad=6)
                else:
                    ax.axis('off')

            plt.tight_layout(pad=1.2)
            st.pyplot(fig)

        # ============================================================
        # 3️⃣ Step 3 — Classification
        # ============================================================
        with st.expander("Step 3: Classification — AI Makes Its Guess"):
            st.markdown("""
            Finally, YOLO transforms all those learned features into a single decision:  
            which digit does it believe your drawing represents?
            """)

            results = model.predict(source=input_array, verbose=False)
            with three_col:
                if results and hasattr(results[0], 'probs'):
                    probs = results[0].probs
                    predicted_class_id = probs.top1
                    confidence = probs.top1conf.item() * 100
                    predicted_label = model.names[predicted_class_id]

                    # --- Display prediction result inside a styled container ---
                    st.markdown("""
                    <div style='
                        background-color:#f5f5f5;
                        border:1px solid #ccc;
                        border-radius:12px;
                        padding:25px 15px;
                        margin-bottom:15px;
                        text-align:center;
                        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
                    '>
                        <h3 style='margin-top:0;'>Prediction Result</h3>
                        <h1 style='font-size:70px; margin:10px 0;'>{predicted_label}</h1>
                        <p style='font-size:18px; color:gray;'>Confidence: {confidence:.2f}%</p>
                    </div>
                    """.format(predicted_label=predicted_label, confidence=confidence),
                    unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please draw a number and click 'Guess the number'.")


            if results and hasattr(results[0], 'probs'):
                probs = results[0].probs
                predicted_class_id = probs.top1
                confidence = probs.top1conf.item() * 100
                predicted_label = model.names[predicted_class_id]

                st.markdown(f"### 🤖 Predicted Number: **{predicted_label}**")
                st.markdown(f"Confidence: **{confidence:.2f}%**")

                prob_df = pd.DataFrame({
                    'Digit': [model.names[i] for i in range(len(model.names))],
                    'Probability': probs.data.cpu().numpy()
                })
                st.bar_chart(prob_df.set_index('Digit'))
            else:
                st.error("The model did not return valid probabilities. Make sure this is a YOLO classification model.")

    else:
        st.warning("Please draw a number first!")
