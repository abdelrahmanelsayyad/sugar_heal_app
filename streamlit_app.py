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

# Enhanced CSS with better responsiveness
st.markdown(f"""
<style>
  /* Base Styles */
  body {{ background-color: {COL['surface']}; color: {COL['text_dark']}; font-family: 'Helvetica Neue', Arial, sans-serif; }}
  
  /* Header Styles */
  .header {{ 
    text-align: center; 
    padding: 20px; 
    background: linear-gradient(135deg, {COL['primary']}, {COL['dark']}); 
    color: {COL['text_light']}; 
    border-radius: 12px; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); 
    margin-bottom: 25px; 
    transition: all 0.3s ease;
  }}
  .header h1 {{ margin:0; font-size:2.2rem; font-weight:600; letter-spacing:1px; }}
  .header p {{ font-size: 1.1rem; margin-top: 8px; opacity: 0.9; }}
  
  /* Instructions Box */
  .instructions {{ 
    background-color: {COL['dark']}; 
    padding: 20px; 
    border-left: 6px solid {COL['accent']}; 
    border-radius: 8px; 
    margin-bottom: 25px; 
    color: {COL['text_light']}; 
    box-shadow: 0 3px 8px rgba(0,0,0,0.2);
  }}
  .instructions strong {{ color:{COL['highlight']}; font-size:1.2rem; }}
  .instructions ol {{ padding-left: 25px; margin-top: 10px; }}
  .instructions li {{ margin-bottom: 5px; }}
  
  /* Logo and Container Styles */
  .logo-container {{
    background-color: {COL['highlight']}; 
    padding: 15px; 
    border-radius: 10px; 
    text-align: center; 
    margin-bottom: 20px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
  }}
  img.logo {{ 
    display: block; 
    margin: 0 auto; 
    width: 100%; 
    max-width: 800px; /* Increased logo size for PC */
    padding: 5px; 
    transition: all 0.3s ease;
  }}
  
  /* Button Styling */
  .stButton>button {{ 
    background: linear-gradient(135deg, {COL['primary']}, {COL['dark']}); 
    color: white; 
    border: none; 
    border-radius: 8px; 
    padding: 12px 28px; 
    font-weight: 500; 
    transition: all .3s ease; 
    box-shadow: 0 3px 8px rgba(0,0,0,0.25); 
    width: 100%;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
  }}
  .stButton>button:hover {{ 
    background: linear-gradient(135deg, {COL['accent']}, {COL['primary']}); 
    transform: translateY(-2px); 
    box-shadow: 0 5px 12px rgba(0,0,0,0.35); 
  }}
  
  /* File Uploader */
  .css-1cpxqw2, [data-testid="stFileUploader"] {{ 
    border: 2px dashed {COL['accent']}; 
    background-color: rgba(59, 108, 83, 0.1); 
    border-radius: 10px; 
    padding: 20px; 
    transition: all 0.3s ease;
  }}
  .css-1cpxqw2:hover, [data-testid="stFileUploader"]:hover {{ 
    border-color: {COL['highlight']}; 
    background-color: rgba(59, 108, 83, 0.2);
  }}
  
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
    height: 100%;  /* Ensure containers are same height */
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .img-container img {{ 
    max-height: 500px; 
    max-width: 100%;
    width: auto !important; 
    margin: 0 auto; 
    display: block; 
    border-radius: 6px;
    transition: all 0.3s ease;
    object-fit: contain;
  }}
  
  /* Image Captions */
  .img-container figcaption, .stImage figcaption, .css-1b0udgb, .css-83jbox {{
    font-size: 1.2rem !important;  /* Larger text for PC */
    color: {COL['text_light']} !important;
    margin-top: 12px !important;
    font-weight: 500 !important;
    text-align: center !important;
  }}
  
  /* Style all Streamlit caption texts */
  figcaption p {{
    font-size: 1.2rem !important;
    margin: 8px 0 !important;
    color: {COL['text_light']} !important;
  }}
  
  /* Guidelines Box */
  .guidelines-box {{ 
    background-color: {COL['dark']}; 
    padding: 18px; 
    border-radius: 10px; 
    color: {COL['text_light']}; 
    margin-bottom: 20px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    border-left: 4px solid {COL['highlight']};
  }}
  .guidelines-box h4 {{ 
    color: {COL['highlight']}; 
    margin-top: 0; 
    font-size: 1.2rem; 
    font-weight: 500;
  }}
  .guidelines-box ul {{ padding-left: .5rem; margin-bottom: 0; list-style-type: none; }}
  .guidelines-box ul li {{ 
    padding-left: 1.5rem; 
    position: relative;
    margin-bottom: 8px;
  }}
  .guidelines-box ul li:before {{ 
    content: "✓"; 
    color: {COL['highlight']};
    position: absolute;
    left: 0;
    font-weight: bold;
  }}
  
  /* Results Section */
  .results-header {{
    text-align: center;
    color: {COL['highlight']};
    margin: 25px 0 15px;
    font-size: 1.8rem;  /* Larger for PC */
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  
  /* Metrics Cards */
  .metric-card {{
    background: linear-gradient(135deg, {COL['dark']}, {COL['accent']});
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
  }}
  .metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.35);
  }}
  .metric-value {{
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 5px;
    color: {COL['text_light']};
  }}
  .metric-label {{
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    font-weight: 500;
  }}
  
  /* Footer */
  .footer {{ 
    text-align: center; 
    padding: 20px 0; 
    margin-top: 40px; 
    border-top: 1px solid {COL['dark']}; 
    color: {COL['light']}; 
    font-size: 1rem; 
  }}
  
  /* Custom Upload Image Size Control */
  .uploaded-image {{
    max-height: 450px;
    width: auto !important;
    object-fit: contain;
  }}
  
  /* Analysis Results Images - NEW */
  .analysis-img {{
    min-height: 350px;
    max-height: 450px;
    width: auto;
    object-fit: contain;
    margin: 0 auto;
    display: block;
  }}
  
  /* Equal Height Columns for Result Images */
  .equal-height-cols .element-container {{
    height: 100%;
  }}
  
  .equal-height-cols [data-testid="column"] {{
    display: flex;
    flex-direction: column;
  }}
  
  .equal-height-cols .stImage {{
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  
  /* Responsive breakpoints */
  /* Mobile Devices */
  @media screen and (max-width: 768px) {{
    .header {{ padding: 15px; }}
    .header h1 {{ font-size: 1.5rem; }}
    .header p {{ font-size: 0.9rem; }}
    .instructions {{ padding: 15px; }}
    .instructions strong {{ font-size: 1rem; }}
    .guidelines-box h4 {{ font-size: 1rem; }}
    .guidelines-box ul {{ font-size: 0.9rem; }}
    .stButton>button {{ padding: 10px 18px; font-size: 1rem; }}
    .img-container img {{ max-height: 300px; }}
    .img-container figcaption, .stImage figcaption, .css-1b0udgb, .css-83jbox {{
      font-size: 0.9rem !important;
    }}
    figcaption p {{ font-size: 0.9rem !important; }}
    .metric-value {{ font-size: 1.5rem; }}
    .metric-label {{ font-size: 0.9rem; }}
    .results-header {{ font-size: 1.3rem; margin: 20px 0 10px; }}
    img.logo {{ max-width: 600px; }}
    .analysis-img {{ min-height: 250px; max-height: 300px; }}
  }}
  
  /* Tablet Devices */
  @media screen and (min-width: 769px) and (max-width: 1024px) {{
    .header h1 {{ font-size: 1.8rem; }}
    .header p {{ font-size: 1rem; }}
    .img-container img {{ max-height: 400px; }}
    .img-container figcaption, .stImage figcaption, .css-1b0udgb, .css-83jbox {{
      font-size: 1.1rem !important;
    }}
    figcaption p {{ font-size: 1.1rem !important; }}
    img.logo {{ max-width: 700px; }}
    .analysis-img {{ min-height: 300px; max-height: 350px; }}
  }}
  
  /* Handle content width based on layout */
  @media screen and (min-width: 1025px) {{
    .content-wrapper {{ max-width: 1200px; margin: 0 auto; }}
    .section-wrapper {{ max-width: 90%; margin: 0 auto; }}
    /* PC-specific text sizes */
    .img-container figcaption, .stImage figcaption, .css-1b0udgb, .css-83jbox {{
      font-size: 1.2rem !important;
    }}
    figcaption p {{ font-size: 1.2rem !important; }}
  }}
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

# ──── Preprocessing & Prediction ─────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return (img_rgb.astype("float32") / 255)[None, ...]

def predict_mask(img_bgr: np.ndarray) -> np.ndarray:
    prob = model.predict(preprocess(img_bgr), verbose=0)[0, ..., 0]
    mask = (prob > THRESHOLD).astype("uint8") * 255
    # Ensure mask is RGB (same dimensions as original) for consistent display
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask

def make_overlay(orig_bgr, mask):
    h, w = orig_bgr.shape[:2]
    # Convert mask to single channel if it's RGB
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask
    
    mask_r = cv2.resize(mask_gray, (w, h), cv2.INTER_NEAREST)
    overlay = orig_bgr.copy()
    overlay[mask_r==255] = (122,164,140)
    return cv2.addWeighted(overlay, ALPHA, orig_bgr, 1-ALPHA, 0)

def calculate_wound_area(mask):
    # Handle both single-channel and RGB masks
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask
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
# Responsive layout that adapts to screen size
col1, col2 = st.columns([2, 1]) 

with col1:
    uploaded = st.file_uploader("Upload wound image", type=["png","jpg","jpeg"])

with col2:
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

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    
    # Responsive image display with controlled size
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    
    # Use consistent parameters for image display
    st.image(pil, caption="Uploaded Wound Image", use_column_width=False, output_format="PNG", 
             clamp=True, channels="RGB")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Full-width button for better mobile tap targets and UX
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    if st.button("Analyze Wound", help="Click to run AI analysis"):
        with st.spinner("Processing wound image..."):
            # Show a more detailed progress bar
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i+1)
                if i==30:
                    mask = predict_mask(orig_bgr)
                if i==70:
                    overlay = make_overlay(orig_bgr, mask)
                    area = calculate_wound_area(mask)
            progress.empty()
        
        st.success("✅ Analysis complete!")
        st.markdown('<div class="results-header">Analysis Results</div>', unsafe_allow_html=True)
        
        # Add CSS class for equal height columns
        st.markdown('<div class="equal-height-cols">', unsafe_allow_html=True)
        
        # Use consistent columns for results
        col1, col2 = st.columns(2)
        
        # Prepare mask for display (ensure it's RGB)
        if len(mask.shape) == 2:
            display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            display_mask = mask
            
        # Ensure both images are resized to the same dimensions for display
        target_height = 400  # Target height for both images
        
        # Calculate aspect ratio and resize
        mask_h, mask_w = display_mask.shape[:2]
        mask_aspect = mask_w / mask_h
        mask_display = cv2.resize(display_mask, (int(target_height * mask_aspect), target_height))
        
        # Prepare overlay
        overlay_h, overlay_w = overlay.shape[:2]
        overlay_aspect = overlay_w / overlay_h
        overlay_display = cv2.resize(overlay, (int(target_height * overlay_aspect), target_height))
        overlay_display = cv2.cvtColor(overlay_display, cv2.COLOR_BGR2RGB)
        
        with col1:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(mask_display, caption="Wound Segmentation Mask", 
                    use_column_width=True, clamp=True, output_format="PNG")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(overlay_display, caption="Segmentation Overlay", 
                    use_column_width=True, clamp=True, output_format="PNG")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True) # Close equal-height-cols
        
        # Enhanced metrics display with custom styling
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
            # Use appropriate mask shape for calculation
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                total_pixels = mask_gray.shape[0] * mask_gray.shape[1]
            else:
                total_pixels = mask.shape[0] * mask.shape[1]
                
            pct = area/total_pixels*100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pct:.2f}%</div>
                <div class="metric-label">Coverage</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──── Footer ───────────────────────────────────────────────────
st.markdown(f"<div class='footer'>© 2025 Sugar Heal AI • Advanced Wound Analysis</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # Close content-wrapper
