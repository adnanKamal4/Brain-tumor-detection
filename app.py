import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image


# Page Configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Background and Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main-header {
        text-align: center;
        color: white;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        text-align: center;
        color: #f0f2f6;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .tumor-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
        color: white;
    }
    
    .prediction-box {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        color: white;
        font-size: 1.5em;
        font-weight: bold;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown('<div class="main-header">🧠 Brain Tumor Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced AI-powered MRI Analysis using DenseNet121 + Grad-CAM</div>', unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_brain_tumor_model():
    MODEL_PATH = r"C:\Users\adnan\OneDrive\Desktop\brain tumour prediction\densenet121_brain_tumor (1).h5"
    return load_model(MODEL_PATH)

model = load_brain_tumor_model()

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# Tumor Info sidebar
with st.sidebar:
    st.markdown("## 📋 Tumor Classifications")
    
    st.markdown("""
    <div class="tumor-card">
        <h4>🔴 Glioma</h4>
        <p>Most common primary brain tumor originating from glial cells. Can be aggressive and require immediate treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tumor-card">
        <h4>🟡 Meningioma</h4>
        <p>Tumors arising from the meninges (protective brain layers). Usually benign and slow-growing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tumor-card">
        <h4>🟢 No Tumor</h4>
        <p>Normal brain MRI showing healthy tissue with no signs of tumor presence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tumor-card">
        <h4>🟣 Pituitary</h4>
        <p>Tumors of the pituitary gland at the brain's base. Can affect hormone production and require specialized treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.markdown("This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis.")


# Preprocessing
def preprocess_image(img):
    img = cv2.resize(img, (224,224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Grad-CAM
def get_gradcam(model, img_array, layer_name="conv5_block16_concat"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = tf.gather(predictions[0], class_idx)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:,:,i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return overlayed

def get_confidence_grade(confidence):
    if confidence >= 0.9:
        return "A+ (Excellent)", "🟢"
    elif confidence >= 0.8:
        return "A (Very Good)", "🟢" 
    elif confidence >= 0.7:
        return "B+ (Good)", "🟡"
    elif confidence >= 0.6:
        return "B (Fair)", "🟡"
    elif confidence >= 0.5:
        return "C (Low)", "🟠"
    else:
        return "D (Very Low)", "🔴"


# Main App Interface
st.markdown("""
<div class="info-box">
    <h3>📤 Upload MRI Image</h3>
    <p>Upload a brain MRI scan in JPG, JPEG, or PNG format. Our AI model will analyze the image and provide:</p>
    <ul>
        <li>🎯 <strong>Tumor Classification</strong> with confidence score</li>
        <li>🔥 <strong>Grad-CAM Heatmap</strong> showing important regions</li>
        <li>📊 <strong>Confidence Grade</strong> indicating prediction reliability</li>
    </ul>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Create columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = preprocess_image(img_rgb)

        # Prediction
        with st.spinner("🔍 Analyzing brain MRI..."):
            preds = model.predict(img_array)
            class_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
            predicted_class = CLASS_NAMES[class_idx]
            
            # Get confidence grade
            grade, grade_emoji = get_confidence_grade(confidence)

        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Diagnosis: {predicted_class.upper()}</h2>
            <h3>📊 Confidence: {confidence*100:.1f}%</h3>
            <h3>{grade_emoji} Grade: {grade}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Generate and display Grad-CAM
        with st.spinner("🔥 Generating Grad-CAM visualization..."):
            heatmap = get_gradcam(model, img_array)
            gradcam_img = overlay_gradcam(img_rgb, heatmap)

        # Display images
        st.markdown("### 📸 Analysis Results")
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.image(img_rgb, caption="📷 Original MRI Scan", use_container_width=True)
        
        with col_img2:
            st.image(gradcam_img, caption="🔥 Grad-CAM Heatmap", use_container_width=True)

        # Additional information based on prediction
        tumor_info = {
            "glioma": {
                "description": "Gliomas are the most common primary brain tumors, originating from glial cells.",
                "urgency": "⚠️ High Priority - Requires immediate medical attention",
                "color": "#FF4444"
            },
            "meningioma": {
                "description": "Meningiomas arise from the meninges and are usually benign.",
                "urgency": "🟡 Moderate Priority - Schedule follow-up consultation",
                "color": "#FFA500"
            },
            "notumor": {
                "description": "No tumor detected - Normal brain tissue observed.",
                "urgency": "✅ Normal Result - Continue regular check-ups",
                "color": "#4CAF50"
            },
            "pituitary": {
                "description": "Pituitary tumors affect the pituitary gland and hormone production.",
                "urgency": "🟣 Specialized Care - Consult endocrinologist",
                "color": "#9C27B0"
            }
        }
        
        info = tumor_info[predicted_class]
        st.markdown(f"""
        <div class="info-box" style="border-left: 4px solid {info['color']};">
            <h3>📋 About {predicted_class.title()}</h3>
            <p><strong>Description:</strong> {info['description']}</p>
            <p><strong>Recommendation:</strong> {info['urgency']}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # sample information when no file is uploaded
    st.markdown("""
    <div class="info-box">
        <h3>🚀 How It Works</h3>
        <ol>
            <li><strong>Upload</strong> your brain MRI image</li>
            <li><strong>AI Analysis</strong> using DenseNet121 deep learning model</li>
            <li><strong>Grad-CAM Visualization</strong> highlights important regions</li>
            <li><strong>Instant Results</strong> with confidence scoring</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Credit Footer
st.markdown(
    """
    <div style='text-align: center; font-size: 12px; color: #cccccc; margin-top: 30px;'>
        Made by <strong>Adnan Kamal (B.Tech AI & DS [MINOR])</strong>
    </div>
    """,
    unsafe_allow_html=True
)
