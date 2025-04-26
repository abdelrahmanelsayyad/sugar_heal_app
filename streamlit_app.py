# streamlit_app.py

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

# Set favicon and responsive layout
LOGO_PATH = Path("GREEN.png")
if LOGO_PATH.exists():
    # Read the image file
    with open(str(LOGO_PATH), "rb") as f:
        favicon_data = f.read()
    # Set favicon using base64 encoded image
    st.set_page_config(
        page_title="Sugar Heal – Wound Analysis",
        page_icon=Image.open(io.BytesIO(favicon_data)),
        layout="wide",  # Wide layout for desktop
        initial_sidebar_state="collapsed"
    )
else:
    st.set_page_config(
        page_title="Sugar Heal – Wound Analysis",
        page_icon="🩹",
        layout="wide",  # Wide layout for desktop
        initial_sidebar_state="collapsed"
    )

# ──── Config ───────────────────────────────────────────────────
MODEL_PATH = Path("unet_wound_segmentation_best.h5")
MODEL_URL  = "https://drive.google.com/uc?id=1_PToBgQjEKAQAZ9ZX10sRpdgxQ18C-18"
LOGO_PATH  = Path("GREEN.png")
IMG_SIZE   = 256
THRESHOLD  = 0.5
ALPHA      = 0.4

# Download model from Google Drive if not present
def download_model():
    if not MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)

# Ensure model is available
download_model()

# ──── Color Palette & CSS ───────────────────────────────────────────────────────
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

# Improved CSS with better responsive design
st.markdown(f"""
<style>
  /* Base styles */
  body {{ background-color: {COL['surface']}; color: {COL['text_dark']}; font-family: 'Helvetica Neue', Arial, sans-serif; }}
  .header {{ text-align: center; padding: 15px; background: linear-gradient(135deg, {COL['primary']}, {COL['dark']}); color: {COL['text_light']}; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px; }}
  .header h1 {{ margin:0; font-size:1.8rem; font-weight:600; letter-spacing:1px; }}
  .header p {{ font-size: 1rem; margin-top: 5px; }}
  .instructions {{ background-color: {COL['dark']}; padding:15px; border-left:6px solid {COL['accent']}; border-radius:8px; margin-bottom:20px; color:{COL['text_light']}; }}
  .instructions strong {{ color:{COL['highlight']}; font-size:1.1rem; }}
  .instructions ol {{ padding-left: 20px; }}
  img.logo {{ display:block; margin:0 auto; width:100%!important; max-width:600px; padding:5px; }}
  .stButton>button {{ background-color:{COL['primary']}; color:white; border:none; border-radius:6px; padding:10px 24px; font-weight:500; transition:all .3s ease; box-shadow:0 2px 5px rgba(0,0,0,.2); width:100%; }}
  .stButton>button:hover {{ background-color:{COL['dark']}; transform:translateY(-2px); box-shadow:0 4px 8px rgba(0,0,0,.3); }}
  .css-1cpxqw2 {{ border:2px dashed {COL['accent']}; background-color:{COL['surface']}; border-radius:8px; padding:15px; }}
  .img-container {{ background-color:{COL['dark']}; padding:10px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,.2); margin-bottom:20px; }}
  .img-container img {{ max-height:400px!important; width:auto!important; margin:0 auto; display:block; }}
  .guidelines-box {{ background-color:{COL['dark']}; padding:15px; border-radius:8px; color:{COL['text_light']}; margin-bottom:15px; }}
  .guidelines-box h4 {{ color:{COL['highlight']}; margin-top:0; font-size:1.1rem; }}
  .guidelines-box ul {{ padding-left: 20px; margin-bottom: 0; }}
  .footer {{ text-align:center; padding:15px 0; margin-top:30px; border-top:1px solid {COL['dark']}; color:{COL['light']}; font-size:.9rem; }}
  
  /* Container styles for desktop/mobile adaptivity */
  .main-container {{
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
  }}
  
  /* Make app container responsive */
  .block-container {{
    max-width: 100%;
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 1rem;
  }}
  
  /* Desktop styles */
  @media screen and (min-width: 992px) {{
    .content-wrapper {{
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
    }}
    .upload-section {{
      flex: 1;
      min-width: 300px;
    }}
    .results-section {{
      flex: 2;
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
    }}
    .result-image {{
      flex: 1;
      min-width: 300px;
    }}
    .header h1 {{ font-size: 2.2rem; }}
    .block-container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
  }}
  
  /* Tablet styles */
  @media screen and (min-width: 768px) and (max-width: 991px) {{
    .header h1 {{ font-size: 1.8rem; }}
  }}
  
  /* Mobile styles */
  @media screen and (max-width: 767px) {{
    .header h1 {{ font-size: 1.5rem; }}
    .header p {{ font-size: 0.9rem; }}
    .instructions strong {{ font-size: 1rem; }}
    .guidelines-box h4 {{ font-size: 1rem; }}
    .guidelines-box ul {{ font-size: 0.9rem; }}
    .stButton>button {{ padding: 8px 16px; }}
    .block-container {{
      padding: 0.5rem;
    }}
  }}
</style>
""", unsafe_allow_html=True)

# Create a responsive container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ──── Header ───────────────────────────────────────────────────
# ──── Header ───────────────────────────────────────────────────
if LOGO_PATH.exists():
    st.markdown(f"""
    <div style="background-color:{COL['highlight']}; padding:10px; border-radius:10px; text-align:center; margin-bottom:15px; overflow:hidden;">
        <img src="data:image/png;base64,{base64.b64encode(open(str(LOGO_PATH), 'rb').read()).decode()}" class="logo" style="border-radius:8px;">
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

# ──── Preprocessing & Prediction ─────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return (img_rgb.astype("float32") / 255)[None, ...]

def predict_mask(img_bgr: np.ndarray) -> np.ndarray:
    prob = model.predict(preprocess(img_bgr), verbose=0)[0, ..., 0]
    return (prob > THRESHOLD).astype("uint8") * 255

def make_overlay(orig_bgr, mask):
    h, w = orig_bgr.shape[:2]
    mask_r = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
    overlay = orig_bgr.copy()
    overlay[mask_r==255] = (122,164,140)
    return cv2.addWeighted(overlay, ALPHA, orig_bgr, 1-ALPHA, 0)

def calculate_wound_area(mask):
    return int(np.sum(mask > 0))

# ──── Instructions ─────────────────────────────────────────────
st.markdown("""
<div class="instructions">
  <strong>📋 How to use this tool:</strong><br>
  <ol>
    <li>Upload a clear wound image (PNG/JPG/JPEG)</li>
    <li>Click <b>Analyze Wound</b></li>
    <li>View segmented mask & overlay</li>
  </ol>
</div>
""", unsafe_allow_html=True)

# ──── Upload & Analysis ────────────────────────────────────────
# Start responsive content wrapper
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload wound image", type=["png","jpg","jpeg"])

# Guidelines box
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

# Close upload section
st.markdown('</div>', unsafe_allow_html=True)

# Results section - will be shown when image is uploaded
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    
    # Show image in its own container
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    # Use container width on mobile, but not on desktop (responsive setting)
    st.image(pil, caption="Uploaded Wound Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Full-width button for better mobile tap targets
    if st.button("Analyze Wound", help="Click to run AI analysis"):
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i+1)
            if i==30:
                mask = predict_mask(orig_bgr)
            if i==70:
                overlay = make_overlay(orig_bgr, mask)
                area = calculate_wound_area(mask)
        progress.empty()
        st.success("Analysis complete!")
        st.markdown(f"<h3 style='text-align:center;color:{COL['highlight']}'>Results</h3>", unsafe_allow_html=True)
        
        # Open results section - will adjust based on screen size
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        
        # First result image
        st.markdown('<div class="result-image">', unsafe_allow_html=True)
        st.image(mask, caption="Mask", clamp=True, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Second result image
        st.markdown('<div class="result-image">', unsafe_allow_html=True)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Close results section
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Metrics in a more compact form - these will stack on mobile automatically
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Wound Area (px)", f"{area:,}")
        with col2:
            pct = area/(mask.shape[0]*mask.shape[1])*100
            st.metric("Coverage", f"{pct:.2f}%")

# Close content wrapper
st.markdown('</div>', unsafe_allow_html=True)

# Close main container 
st.markdown('</div>', unsafe_allow_html=True)

# ──── Footer ───────────────────────────────────────────────────
st.markdown(f"<div class='footer'>© 2025 Sugar Heal AI</div>", unsafe_allow_html=True)
