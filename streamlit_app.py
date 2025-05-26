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
import torch
import gdown
import segmentation_models_pytorch as smp

st.set_page_config(
    page_title="Sugar Heal – Advanced Wound Analysis",
    page_icon="🩹",
    layout="wide",  # Use wide layout for PC
    initial_sidebar_state="collapsed"
)

# ──── Config ───────────────────────────────────────────────────
# Original Sugar Heal Model (Wound Hole Segmentation)
SUGAR_MODEL_PATH = Path("unet_wound_segmentation_best.h5")
SUGAR_MODEL_URL  = "https://drive.google.com/uc?id=1_PToBgQjEKAQAZ9ZX10sRpdgxQ18C-18"

# Advanced Tissue Analysis Model
TISSUE_MODEL_PATH = Path("best_model_streamlit.pth")
TISSUE_MODEL_ID = "1q0xk9wll0eyF3-CKEc5s6MfG0gE_jde1"

LOGO_PATH  = Path("GREEN.png")
IMG_SIZE   = 256
THRESHOLD  = 0.5
ALPHA      = 0.4

# Tissue Analysis Config
N_CLASSES = 9
ENCODER = "mit_b3"
CLASS_NAMES = [
    "background", "fibrin", "granulation", "callus", "necrotic", "eschar", "neodermis", "tendon", "dressing"
]
PALETTE = [
    (0, 0, 0),         # 0: background (Black)
    (255, 255, 0),     # 1: fibrin (Yellow)
    (255, 0, 0),       # 2: granulation (Red)
    (0, 0, 255),       # 3: callus (Blue)
    (255, 165, 0),     # 4: necrotic (Orange)
    (128, 0, 128),     # 5: eschar (Purple)
    (0, 255, 255),     # 6: neodermis (Cyan)
    (255, 192, 203),   # 7: tendon (Pink)
    (0, 255, 0),       # 8: dressing (Green)
]

# Tissue health scoring weights
TISSUE_HEALTH_WEIGHTS = {
    "granulation": 0.8,    # Good healing tissue
    "neodermis": 0.9,      # Excellent - new skin
    "fibrin": 0.6,         # Moderate - part of healing
    "callus": 0.4,         # Poor - hard tissue
    "necrotic": -0.8,      # Very bad - dead tissue
    "eschar": -0.6,        # Bad - scab tissue
    "tendon": 0.2,         # Neutral - exposed but not necessarily bad
    "dressing": 0.0,       # Neutral
    "background": 0.0      # Neutral
}

# Download models from Google Drive if not present
def download_models():
    if not SUGAR_MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        st.info("Downloading Sugar Heal segmentation model...")
        gdown.download(SUGAR_MODEL_URL, str(SUGAR_MODEL_PATH), quiet=False)
    
    if not TISSUE_MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        st.info("Downloading tissue analysis model...")
        gdown.download(f"https://drive.google.com/uc?id={TISSUE_MODEL_ID}", str(TISSUE_MODEL_PATH), quiet=False)

# Ensure models are available
download_models()

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
    "success"    : "#28a745",
    "warning"    : "#ffc107",
    "danger"     : "#dc3545",
}

# Enhanced CSS with better responsiveness and new features
st.markdown(f"""
<style>
  /* Base Styles */
  body {{ background-color: {COL['surface']}; color: {COL['text_dark']}; font-family: 'Helvetica Neue', Arial, sans-serif; }}
  
  /* Header Styles */
  .header {{ 
    text-align: center; 
    padding: 25px; 
    background: linear-gradient(135deg, {COL['primary']}, {COL['dark']}); 
    color: {COL['text_light']}; 
    border-radius: 15px; 
    box-shadow: 0 6px 20px rgba(0,0,0,0.4); 
    margin-bottom: 30px; 
    transition: all 0.3s ease;
  }}
  .header h1 {{ margin:0; font-size:2.5rem; font-weight:700; letter-spacing:1.5px; }}
  .header p {{ font-size: 1.2rem; margin-top: 10px; opacity: 0.95; }}
  
  /* Instructions Box */
  .instructions {{ 
    background-color: {COL['dark']}; 
    padding: 25px; 
    border-left: 8px solid {COL['accent']}; 
    border-radius: 10px; 
    margin-bottom: 30px; 
    color: {COL['text_light']}; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
  .instructions strong {{ color:{COL['highlight']}; font-size:1.3rem; }}
  .instructions ol {{ padding-left: 30px; margin-top: 15px; }}
  .instructions li {{ margin-bottom: 8px; font-size: 1.1rem; }}
  
  /* Logo and Container Styles */
  .logo-container {{
    background-color: {COL['highlight']}; 
    padding: 20px; 
    border-radius: 12px; 
    text-align: center; 
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
  }}
  img.logo {{ 
    display: block; 
    margin: 0 auto; 
    width: 100%; 
    max-width: 900px;
    padding: 8px; 
    transition: all 0.3s ease;
  }}
  
  /* Button Styling */
  .stButton>button {{ 
    background: linear-gradient(135deg, {COL['primary']}, {COL['dark']}); 
    color: white; 
    border: none; 
    border-radius: 10px; 
    padding: 15px 35px; 
    font-weight: 600; 
    transition: all .3s ease; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); 
    width: 100%;
    font-size: 1.2rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
  }}
  .stButton>button:hover {{ 
    background: linear-gradient(135deg, {COL['accent']}, {COL['primary']}); 
    transform: translateY(-3px); 
    box-shadow: 0 6px 18px rgba(0,0,0,0.4); 
  }}
  
  /* Analysis Mode Toggle */
  .analysis-mode {{
    background: linear-gradient(135deg, {COL['dark']}, {COL['accent']});
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
    text-align: center;
    color: {COL['text_light']};
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  }}
  .analysis-mode h3 {{
    margin: 0 0 15px 0;
    color: {COL['highlight']};
    font-size: 1.4rem;
  }}
  
  /* File Uploader */
  .css-1cpxqw2, [data-testid="stFileUploader"] {{ 
    border: 3px dashed {COL['accent']}; 
    background-color: rgba(59, 108, 83, 0.15); 
    border-radius: 12px; 
    padding: 25px; 
    transition: all 0.3s ease;
  }}
  .css-1cpxqw2:hover, [data-testid="stFileUploader"]:hover {{ 
    border-color: {COL['highlight']}; 
    background-color: rgba(59, 108, 83, 0.25);
  }}
  
  /* Image Container */
  .img-container {{ 
    background-color: {COL['dark']}; 
    padding: 25px; 
    border-radius: 15px; 
    box-shadow: 0 5px 15px rgba(0,0,0,0.35); 
    margin-bottom: 10px; 
    transition: all 0.3s ease;
    overflow: hidden;
    text-align: center;
    height: 100%;
    display: flex !important;
    flex-direction: column;
    justify-content: center;
    align-items: center !important;
    width: 100% !important;
  }}
  
  /* Make images larger and centered */
  .img-container img,
  .stImage img {{ 
    max-height: 700px !important;
    max-width: 100% !important;
    width: auto !important; 
    height: auto !important;
    margin: 0 auto !important; 
    display: block !important; 
    border-radius: 8px !important;
    transition: all 0.3s ease;
    object-fit: contain !important;
  }}
  
  /* Center all Streamlit images */
  [data-testid="stImage"] {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
  }}
  
  /* Image Captions */
  .img-container figcaption, .stImage figcaption, .css-1b0udgb, .css-83jbox {{
    font-size: 1.3rem !important;
    color: {COL['text_light']} !important;
    margin-top: 15px !important;
    font-weight: 600 !important;
    text-align: center !important;
    width: 100% !important;
  }}
  
  figcaption p {{
    font-size: 1.3rem !important;
    margin: 10px 0 !important;
    color: {COL['text_light']} !important;
    text-align: center !important;
  }}

  /* Guidelines Box */
  .guidelines-box {{ 
    background-color: {COL['dark']}; 
    padding: 20px; 
    border-radius: 12px; 
    color: {COL['text_light']}; 
    margin-bottom: 25px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    border-left: 5px solid {COL['highlight']};
  }}
  .guidelines-box h4 {{ 
    color: {COL['highlight']}; 
    margin-top: 0; 
    font-size: 1.3rem; 
    font-weight: 600;
  }}
  .guidelines-box ul {{ padding-left: 0.8rem; margin-bottom: 0; list-style-type: none; }}
  .guidelines-box ul li {{ 
    padding-left: 2rem; 
    position: relative;
    margin-bottom: 10px;
    font-size: 1.1rem;
  }}
  .guidelines-box ul li:before {{ 
    content: "✓"; 
    color: {COL['highlight']};
    position: absolute;
    left: 0;
    font-weight: bold;
    font-size: 1.2rem;
  }}
  
  /* Results Section */
  .results-header {{
    text-align: center;
    color: {COL['highlight']};
    margin: 30px 0 20px;
    font-size: 2rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
  }}
  
  /* Metrics Cards */
  .metric-card {{
    background: linear-gradient(135deg, {COL['dark']}, {COL['accent']});
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
    margin-bottom: 15px;
  }}
  .metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
  }}
  .metric-value {{
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 8px;
    color: {COL['text_light']};
  }}
  .metric-label {{
    font-size: 1.1rem;
    color: rgba(255,255,255,0.9);
    font-weight: 600;
  }}
  
  /* Health Score Styling */
  .health-excellent {{ background: linear-gradient(135deg, {COL['success']}, #20c997); }}
  .health-good {{ background: linear-gradient(135deg, #28a745, #20c997); }}
  .health-moderate {{ background: linear-gradient(135deg, {COL['warning']}, #fd7e14); }}
  .health-poor {{ background: linear-gradient(135deg, {COL['danger']}, #e74c3c); }}
  
  /* Tissue Composition */
  .tissue-item {{
    background-color: {COL['dark']};
    padding: 12px 18px;
    margin: 8px 0;
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }}
  .tissue-name {{
    font-weight: 600;
    font-size: 1.1rem;
    text-transform: capitalize;
  }}
  .tissue-percent {{
    font-weight: 700;
    font-size: 1.2rem;
    color: {COL['highlight']};
  }}
  
  /* Analysis Tabs */
  .analysis-tab {{
    background-color: {COL['dark']};
    border-radius: 12px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  }}
  .tab-title {{
    color: {COL['highlight']};
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 20px;
    text-align: center;
    border-bottom: 2px solid {COL['accent']};
    padding-bottom: 10px;
  }}
  
  /* Footer */
  .footer {{ 
    text-align: center; 
    padding: 25px 0; 
    margin-top: 50px; 
    border-top: 2px solid {COL['dark']}; 
    color: {COL['light']}; 
    font-size: 1.1rem; 
    font-weight: 500;
  }}
  
  .section-wrapper{{
    width:100%;
    display:flex;
    align-items:center;
    justify-content:center;
  }}
  
  /* Responsive breakpoints */
  @media screen and (max-width: 768px) {{
    .header {{ padding: 20px; }}
    .header h1 {{ font-size: 1.8rem; }}
    .header p {{ font-size: 1rem; }}
    .instructions {{ padding: 20px; }}
    .img-container img, .stImage img {{ max-height: 500px !important; }}
    .metric-value {{ font-size: 1.6rem; }}
    .results-header {{ font-size: 1.5rem; }}
  }}
  
  @media screen and (min-width: 769px) and (max-width: 1024px) {{
    .header h1 {{ font-size: 2.2rem; }}
    .img-container img, .stImage img {{ max-height: 600px !important; }}
  }}
  
  @media screen and (min-width: 1025px) {{
    .content-wrapper {{ max-width: 1400px; margin: 0 auto; }}
    .section-wrapper {{ max-width: 95%; margin: 0 auto; }}
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
  <h1>Sugar Heal – Advanced Wound Analysis</h1>
  <p>Dual AI-powered system: Precise wound segmentation + Advanced tissue composition analysis</p>
</div>
""", unsafe_allow_html=True)

# ──── Sugar Heal Model Functions ────────────────────────────────────────────────
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
def load_sugar_model():
    with st.spinner("Loading Sugar Heal segmentation model..."):
        return tf.keras.models.load_model(
            str(SUGAR_MODEL_PATH),
            custom_objects={"dice_coefficient": dice_coefficient, "iou_metric": iou_metric},
            compile=False
        )

def preprocess_sugar(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return (img_rgb.astype("float32") / 255)[None, ...]

def predict_wound_mask(img_bgr: np.ndarray, model) -> np.ndarray:
    prob = model.predict(preprocess_sugar(img_bgr), verbose=0)[0, ..., 0]
    mask = (prob > THRESHOLD).astype("uint8") * 255
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask

def make_overlay(orig_bgr, mask):
    h, w = orig_bgr.shape[:2]
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask
    
    mask_r = cv2.resize(mask_gray, (w, h), cv2.INTER_NEAREST)
    overlay = orig_bgr.copy()
    overlay[mask_r==255] = (122,164,140)
    return cv2.addWeighted(overlay, ALPHA, orig_bgr, 1-ALPHA, 0)

def calculate_wound_area(mask):
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask
    return int(np.sum(mask_gray > 0))

# ──── Tissue Analysis Model Functions ────────────────────────────────────────────
@st.cache_resource
def load_tissue_model():
    with st.spinner("Loading tissue analysis model..."):
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=N_CLASSES,
            decoder_attention_type='scse',
            activation=None,
        )
        state_dict = torch.load(str(TISSUE_MODEL_PATH), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

def preprocess_tissue(image_pil):
    image = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

def postprocess_tissue(mask):
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    mask = mask.argmax(0).cpu().numpy()
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        color_mask[mask == idx] = color
    return color_mask, mask

def calculate_tissue_percentages(mask, class_names):
    total_pixels = mask.size
    percentages = {}
    for idx, name in enumerate(class_names):
        class_pixels = np.sum(mask == idx)
        if class_pixels > 0:
            percentages[name] = (class_pixels / total_pixels) * 100
    return percentages

def calculate_wound_health_score(tissue_percentages):
    """Calculate overall wound health score based on tissue composition"""
    score = 0
    total_weight = 0
    
    for tissue, percentage in tissue_percentages.items():
        if tissue in TISSUE_HEALTH_WEIGHTS:
            weight = TISSUE_HEALTH_WEIGHTS[tissue]
            score += weight * (percentage / 100)
            total_weight += abs(weight) * (percentage / 100)
    
    if total_weight > 0:
        normalized_score = (score + total_weight) / (2 * total_weight) * 100
        return max(0, min(100, normalized_score))
    return 50  # Neutral score if no tissue detected

def get_health_status(score):
    """Convert health score to status and color class"""
    if score >= 80:
        return "Excellent", "health-excellent"
    elif score >= 65:
        return "Good", "health-good"
    elif score >= 40:
        return "Moderate", "health-moderate"
    else:
        return "Poor", "health-poor"

def generate_recommendations(tissue_percentages, health_score):
    """Generate healing recommendations based on tissue analysis"""
    recommendations = []
    
    if tissue_percentages.get("necrotic", 0) > 5:
        recommendations.append("⚠️ Debridement recommended - significant necrotic tissue present")
    
    if tissue_percentages.get("eschar", 0) > 10:
        recommendations.append("🧹 Consider eschar removal for better healing")
    
    if tissue_percentages.get("granulation", 0) > 40:
        recommendations.append("✅ Good granulation tissue - wound healing well")
    
    if tissue_percentages.get("neodermis", 0) > 0:
        recommendations.append("🌟 New skin formation detected - excellent progress")
    
    if health_score < 40:
        recommendations.append("🩺 Consult healthcare provider - wound requires attention")
    elif health_score > 80:
        recommendations.append("🎉 Wound showing excellent healing progress")
    
    if tissue_percentages.get("fibrin", 0) > 20:
        recommendations.append("💧 Maintain moist wound environment")
    
    return recommendations if recommendations else ["📋 Continue current wound care regimen"]

# ──── Load Models ────────────────────────────────────────────────
try:
    sugar_model = load_sugar_model()
    tissue_model = load_tissue_model()
    st.success("✅ Both AI models loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ──── Instructions ─────────────────────────────────────────────
st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
st.markdown("""
<div class="instructions">
  <strong>🔬 Advanced Wound Analysis System:</strong><br>
  <ol>
    <li><b>Upload</b> a clear wound image (PNG/JPG/JPEG)</li>
    <li><b>Choose</b> analysis mode: Basic segmentation or Complete analysis</li>
    <li><b>Analyze</b> to get precise wound boundaries + detailed tissue composition</li>
    <li><b>View</b> comprehensive results with health scoring and healing recommendations</li>
  </ol>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──── Analysis Mode Selection ────────────────────────────────────────────────
st.markdown("""
<div class="analysis-mode">
  <h3>🎯 Analysis Mode</h3>
</div>
""", unsafe_allow_html=True)

analysis_mode = st.radio(
    "Select analysis type:",
    ["🔍 Basic Segmentation (Fast)", "🧬 Complete Analysis (Detailed)"],
    help="Basic: Wound boundary detection only. Complete: Full tissue analysis + health scoring"
)

# ──── Upload & Analysis ────────────────────────────────────────
col1, col2 = st.columns([2, 1]) 

with col1:
    uploaded = st.file_uploader("Upload wound image", type=["png","jpg","jpeg"])

with col2:
    st.markdown("""
    <div class="guidelines-box">
        <h4>📸 Image Guidelines</h4>
        <ul>
            <li>Good lighting & focus</li>
            <li>Wound clearly visible</li>
            <li>Consistent distance</li>
            <li>Include reference scale</li>
            <li>Clean wound area</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    
    # Display uploaded image
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(pil, caption="Uploaded Wound Image", use_container_width=True, 
             output_format="PNG", clamp=True, channels="RGB")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    if st.button("🚀 Analyze Wound", help="Click to run AI analysis"):
        
        if "Basic" in analysis_mode:
            # ──── Basic Segmentation Analysis ────────────────────────────────────
            with st.spinner("Running basic wound segmentation..."):
                progress = st.progress(0)
                for i in range(100):
                    progress.progress(i+1)
                    if i==50:
                        wound_mask = predict_wound_mask(orig_bgr, sugar_model)
                        overlay = make_overlay(orig_bgr, wound_mask)
                        area = calculate_wound_area(wound_mask)
                progress.empty()
            
            st.success("✅ Basic analysis complete!")
            st.markdown('<div class="results-header">Wound Segmentation Results</div>', unsafe_allow_html=True)
            
            # Display results
            col1, col2 = st.columns(2)
            
            # Prepare images for display
            if len(wound_mask.shape) == 2:
                display_mask = cv2.cvtColor(wound_mask, cv2.COLOR_GRAY2RGB)
            else:
                display_mask = wound_mask
            
            overlay_display = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
            with col1:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(display_mask, caption="Wound Boundary Mask", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(overlay_display, caption="Segmentation Overlay", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Basic metrics
            st.markdown("<h3 style='text-align:center;margin-top:20px;margin-bottom:20px;'>Basic Wound Metrics</h3>", unsafe_allow_html=True)
            
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{area:,}</div>
                    <div class="metric-label">Wound Area (pixels)</div>
                </div>
                """, unsafe_allow_html=True)
                
            with metric_col2:
                if len(wound_mask.shape) == 3:
                    mask_gray = cv2.cvtColor(wound_mask, cv2.COLOR_RGB2GRAY)
                    total_pixels = mask_gray.shape[0] * mask_gray.shape[1]
                else:
                    total_pixels = wound_mask.shape[0] * wound_mask.shape[1]
                
                wound_percentage = (area / total_pixels) * 100
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{wound_percentage:.1f}%</div>
                    <div class="metric-label">Wound Coverage</div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # ──── Complete Analysis (Both Models) ────────────────────────────────────
            with st.spinner("Running complete wound analysis..."):
                progress = st.progress(0)
                
                # Step 1: Sugar Heal segmentation
                for i in range(25):
                    progress.progress(i+1)
                wound_mask = predict_wound_mask(orig_bgr, sugar_model)
                overlay = make_overlay(orig_bgr, wound_mask)
                area = calculate_wound_area(wound_mask)
                
                # Step 2: Tissue analysis
                for i in range(25, 75):
                    progress.progress(i+1)
                
                with torch.no_grad():
                    tissue_input = preprocess_tissue(pil)
                    tissue_output = tissue_model(tissue_input)
                    tissue_mask_colored, tissue_mask_raw = postprocess_tissue(tissue_output)
                
                # Step 3: Analysis calculations
                for i in range(75, 100):
                    progress.progress(i+1)
                
                tissue_percentages = calculate_tissue_percentages(tissue_mask_raw, CLASS_NAMES)
                health_score = calculate_wound_health_score(tissue_percentages)
                health_status, health_class = get_health_status(health_score)
                recommendations = generate_recommendations(tissue_percentages, health_score)
                
                progress.empty()
            
            st.success("✅ Complete analysis finished!")
            st.markdown('<div class="results-header">Complete Wound Analysis Results</div>', unsafe_allow_html=True)
            
            # ──── Results Display Section ────────────────────────────────────────
            
            # Image Results Grid
            st.markdown("<h3 style='text-align:center;margin:20px 0;'>Visual Analysis</h3>", unsafe_allow_html=True)
            
            img_col1, img_col2 = st.columns(2)
            
            # Prepare images for display
            if len(wound_mask.shape) == 2:
                display_mask = cv2.cvtColor(wound_mask, cv2.COLOR_GRAY2RGB)
            else:
                display_mask = wound_mask
            
            overlay_display = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            tissue_display = cv2.cvtColor(tissue_mask_colored, cv2.COLOR_BGR2RGB)
            
            with img_col1:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(display_mask, caption="Wound Boundary Detection", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(tissue_display, caption="Tissue Composition Analysis", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with img_col2:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(overlay_display, caption="Segmentation Overlay", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Tissue Legend
                st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
                st.markdown('<div class="tab-title">Tissue Color Legend</div>', unsafe_allow_html=True)
                
                legend_html = ""
                for idx, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
                    if name in tissue_percentages and tissue_percentages[name] > 0.1:
                        color_hex = f"rgb({color[0]}, {color[1]}, {color[2]})"
                        legend_html += f"""
                        <div style="display: flex; align-items: center; margin: 8px 0;">
                            <div style="width: 20px; height: 20px; background-color: {color_hex}; 
                                      border-radius: 4px; margin-right: 10px; border: 1px solid #ccc;"></div>
                            <span style="text-transform: capitalize; font-weight: 600;">{name}</span>
                        </div>
                        """
                
                st.markdown(legend_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Health Score Section
            st.markdown("<h3 style='text-align:center;margin:30px 0 20px;'>Wound Health Assessment</h3>", unsafe_allow_html=True)
            
            health_col1, health_col2, health_col3 = st.columns([1, 2, 1])
            
            with health_col1:
                if len(wound_mask.shape) == 3:
                    mask_gray = cv2.cvtColor(wound_mask, cv2.COLOR_RGB2GRAY)
                    total_pixels = mask_gray.shape[0] * mask_gray.shape[1]
                else:
                    total_pixels = wound_mask.shape[0] * wound_mask.shape[1]
                
                wound_percentage = (area / total_pixels) * 100
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{area:,}</div>
                    <div class="metric-label">Wound Area (pixels)</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{wound_percentage:.1f}%</div>
                    <div class="metric-label">Wound Coverage</div>
                </div>
                """, unsafe_allow_html=True)
            
            with health_col2:
                st.markdown(f"""
                <div class="metric-card {health_class}">
                    <div class="metric-value">{health_score:.0f}/100</div>
                    <div class="metric-label">Wound Health Score</div>
                    <div style="margin-top: 10px; font-size: 1.3rem; font-weight: 700;">
                        Status: {health_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with health_col3:
                dominant_tissue = max(tissue_percentages.items(), key=lambda x: x[1])
                tissue_count = len([t for t in tissue_percentages.values() if t > 1])
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{tissue_count}</div>
                    <div class="metric-label">Tissue Types Detected</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dominant_tissue[1]:.1f}%</div>
                    <div class="metric-label">Dominant: {dominant_tissue[0].title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Tissue Composition
            st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
            st.markdown('<div class="tab-title">Detailed Tissue Composition</div>', unsafe_allow_html=True)
            
            # Sort tissues by percentage
            sorted_tissues = sorted(tissue_percentages.items(), key=lambda x: x[1], reverse=True)
            
            for tissue_name, percentage in sorted_tissues:
                if percentage > 0.1:  # Only show tissues with >0.1% presence
                    st.markdown(f"""
                    <div class="tissue-item">
                        <span class="tissue-name">{tissue_name.title()}</span>
                        <span class="tissue-percent">{percentage:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations Section
            st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
            st.markdown('<div class="tab-title">Healing Recommendations</div>', unsafe_allow_html=True)
            
            for recommendation in recommendations:
                st.markdown(f"""
                <div style="background-color: rgba(122,164,140,0.1); padding: 12px; margin: 8px 0; 
                           border-left: 4px solid rgb(122,164,140); border-radius: 6px;">
                    <span style="font-size: 1.1rem; color: #E0E0E0;">{recommendation}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clinical Notes Section
            st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
            st.markdown('<div class="tab-title">Clinical Insights</div>', unsafe_allow_html=True)
            
            clinical_notes = []
            
            if tissue_percentages.get("granulation", 0) > 50:
                clinical_notes.append("🔬 High granulation tissue indicates active healing phase")
            
            if tissue_percentages.get("necrotic", 0) > 10:
                clinical_notes.append("⚠️ Significant necrotic tissue may impede healing process")
            
            if tissue_percentages.get("neodermis", 0) > 5:
                clinical_notes.append("🌱 New skin formation suggests wound is in proliferative phase")
            
            if health_score > 75:
                clinical_notes.append("📈 Wound shows strong healing trajectory")
            elif health_score < 35:
                clinical_notes.append("📉 Wound may benefit from revised treatment approach")
            
            if not clinical_notes:
                clinical_notes.append("📋 Wound appears to be progressing normally")
            
            for note in clinical_notes:
                st.markdown(f"""
                <div style="background-color: rgba(59,108,83,0.15); padding: 15px; margin: 10px 0; 
                           border-radius: 8px; border: 1px solid rgba(59,108,83,0.3);">
                    <span style="font-size: 1.1rem; color: #E0E0E0; font-weight: 500;">{note}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ──── Footer ────────────────────────────────────────────────────
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2024 Sugar Heal - Advanced Wound Analysis System | Powered by AI for Better Healthcare", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close content-wrapper
