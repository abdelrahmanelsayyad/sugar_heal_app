# streamlit_app.py
# NOTE: Only the lines explicitly marked >>> were changed to make
#       the two result images identical in width/length
#       and to keep them responsive on mobile.

import io
import os
import sys
import base64
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K

st.set_page_config(
    page_title="Sugar Heal – Wound Analysis",
    page_icon="🩹",
    layout="wide",  # Use wide layout for PC
    initial_sidebar_state="collapsed"
)

# ──── Config ───────────────────────────────────────────────────
MODEL_PATH = Path("unet_wound_segmentation_best.h5")
MODEL_URL  = "https://drive.google.com/uc?id=1_PToBgQjEKAQAZ9ZX10sRpdgxQ18C-18"
LOGO_PATH  = Path("GREEN.png")
IMG_SIZE   = 256
THRESHOLD  = 0.5
ALPHA      = 0.4

def download_model():
    if not MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)

download_model()

# ──── Color Palette & CSS ──────────────────────────────────────
COL = {
    "primary"    : "#074225",
    "secondary"  : "#41706F",
    "accent"     : "#3B6C53",
    "dark"       : "#335F4B",
    "light"      : "#81A295",
    "surface"    : "#202020",
    "text_dark"  : "#E0E0E0",
    "text_light" : "#FFFFFF",
    "highlight"  : "rgb(122,164,140)",
}

st.markdown(f"""
<style>
  /* --- existing styles remain unchanged --- */

  /* Image Container */
  .img-container {{
    background-color: {COL['dark']};
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    margin-bottom: 25px;
    transition: all 0.3s ease;
    overflow: hidden;
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .img-container img {{
    max-height: 500px;
    max-width: 100%;
    width: 100% !important;        /* >>> force equal width */
    margin: 0 auto;
    display: block;
    border-radius: 6px;
    transition: all 0.3s ease;
    object-fit: contain;
  }}

  /* --- rest of the original CSS is unchanged --- */
</style>
""", unsafe_allow_html=True)

# ──── Page Layout ────────────────────────────────────────────────
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# ──── Header ───────────────────────────────────────────────────
if LOGO_PATH.exists():
    st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{base64.b64encode(open(str(LOGO_PATH), 'rb').read()).decode()}" class="logo">
    </div>
    """, unsafe_allow_html=True)
st.markdown("""
<div class="header">
  <h1>Sugar Heal – Wound Analysis</h1>
  <p>AI-powered wound segmentation for better healing outcomes</p>
</div>
""", unsafe_allow_html=True)

# ──── Metrics ──────────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), "float32")
    y_pred_f = K.cast(K.flatten(y_pred), "float32")
    inter    = K.sum(y_true_f * y_pred_f)
    return (2*inter + smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), "float32")
    y_pred_f = K.cast(K.flatten(y_pred), "float32")
    inter    = K.sum(y_true_f * y_pred_f)
    union    = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)

@st.cache_resource
def load_model():
    with st.spinner("Loading AI model..."):
        return tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={"dice_coefficient": dice_coefficient, "iou_metric": iou_metric},
            compile=False
        )

try:
    model = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ──── Preprocessing & Prediction ───────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return (img_rgb.astype("float32") / 255)[None, ...]

def predict_mask(img_bgr: np.ndarray) -> np.ndarray:
    prob = model.predict(preprocess(img_bgr), verbose=0)[0, ..., 0]
    mask = (prob > THRESHOLD).astype("uint8") * 255
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask

def make_overlay(orig_bgr, mask):
    h, w = orig_bgr.shape[:2]
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if len(mask.shape) == 3 else mask
    mask_r = cv2.resize(mask_gray, (w, h), cv2.INTER_NEAREST)
    overlay = orig_bgr.copy()
    overlay[mask_r == 255] = (122, 164, 140)
    return cv2.addWeighted(overlay, ALPHA, orig_bgr, 1 - ALPHA, 0)

def calculate_wound_area(mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if len(mask.shape) == 3 else mask
    return int(np.sum(mask_gray > 0))

# ──── Instructions ─────────────────────────────────────────────
st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
st.markdown("""
<div class="instructions">
  <strong>📋 How to use this tool:</strong><br>
  <ol>
    <li>Upload a clear wound image (PNG/JPG/JPEG)</li>
    <li>Click <b>Analyze Wound</b></li>
    <li>View segmented mask & overlay with detailed metrics</li>
  </ol>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──── Upload & Analysis ────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload wound image", type=["png", "jpg", "jpeg"])

with col2:
    st.markdown("""
    <div class="guidelines-box">
        <h4>📸 Image Guidelines</h4>
        <ul>
            <li>Good lighting</li>
            <li>Wound clearly visible</li>
            <li>Consistent distance</li>
            <li>Include reference scale</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(pil, caption="Uploaded Wound Image", use_container_width=False, output_format="PNG", clamp=True, channels="RGB")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    if st.button("Analyze Wound", help="Click to run AI analysis"):
        with st.spinner("Processing wound image..."):
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)
                if i == 30:
                    mask = predict_mask(orig_bgr)
                if i == 70:
                    overlay = make_overlay(orig_bgr, mask)
                    area = calculate_wound_area(mask)
            progress.empty()

        st.success("✅ Analysis complete!")
        st.markdown('<div class="results-header">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="equal-height-cols">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        if len(mask.shape) == 2:
            display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            display_mask = mask

        target_height = 400
        mask_h, mask_w = display_mask.shape[:2]
        mask_aspect = mask_w / mask_h
        mask_display = cv2.resize(display_mask, (int(target_height * mask_aspect), target_height))

        overlay_h, overlay_w = overlay.shape[:2]
        overlay_aspect = overlay_w / overlay_h
        overlay_display = cv2.resize(overlay, (int(target_height * overlay_aspect), target_height))
        overlay_display = cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB)

        with col1:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(
                mask_display,
                caption="Wound Segmentation Mask",
                use_container_width=True,   # >>> changed to True
                clamp=True,
                output_format="PNG"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(
                overlay_display,
                caption="Segmentation Overlay",
                use_container_width=True,   # >>> changed to True
                clamp=True,
                output_format="PNG"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close equal-height-cols

        st.markdown("<h3 style='text-align:center;margin-top:20px;margin-bottom:15px;font-size:1.5rem;'>Wound Metrics</h3>", unsafe_allow_html=True)

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{area:,}</div>
                <div class="metric-label">Wound Area (pixels)</div>
            </div>
            """, unsafe_allow_html=True)

        with metric_col2:
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                total_pixels = mask_gray.shape[0] * mask_gray.shape[1]
            else:
                total_pixels = mask.shape[0] * mask.shape[1]

            pct = area / total_pixels * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pct:.2f}%</div>
                <div class="metric-label">Coverage</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──── Footer ───────────────────────────────────────────────────
st.markdown(f"<div class='footer'>© 2025 Sugar Heal AI • Advanced Wound Analysis</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close content-wrapper
