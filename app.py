import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Student Attention Detection",
    page_icon="📸",
    layout="wide"
)

# --- Classes ---
classes = ['Attentive', 'Distracted', 'Sleepy', 'Bullying', 'Daydreaming', 'Hand_raising', 'Phone_use']

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Using device: {device}")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = "emotion_model_2.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please upload the model file.")
        return None
    
    try:
        model = models.resnet18(pretrained=False)  # Don't download pretrained weights
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# --- Prediction Function ---
def predict_image(model, image):
    """Predict student behavior from image"""
    try:
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred = torch.argmax(probs).item()
        
        return classes[pred], probs[pred].item()
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0

# --- Main App ---
def main():
    st.title("📸 Student Attention Detection System")
    st.markdown("Upload an image or use webcam to classify student behavior in real time.")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar options
    st.sidebar.header("Options")
    mode = st.sidebar.radio("Choose input method:", ["Upload Image", "Webcam Feed"])
    
    if mode == "Upload Image":
        # File upload mode
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Prediction")
                
                # Make prediction
                with st.spinner("Analyzing..."):
                    pred_class, confidence = predict_image(model, image)
                
                # Display results
                st.success(f"**Predicted Class:** {pred_class}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Progress bar for confidence
                st.progress(confidence)
    
    elif mode == "Webcam Feed":
        st.subheader("Real-time Webcam Detection")
        
        # Webcam controls
        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")
        
        # Placeholders for video and predictions
        frame_placeholder = st.empty()
        prediction_placeholder = st.empty()
        
        if start_button:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. This might not work in cloud environments.")
                st.info("💡 Try uploading an image instead!")
            else:
                st.success("📹 Webcam started successfully!")
                
                # Session state for controlling the loop
                if 'webcam_running' not in st.session_state:
                    st.session_state.webcam_running = True
                
                try:
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Failed to grab frame from webcam.")
                            break
                        
                        # Convert frame to RGB for PIL
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        
                        # Make prediction
                        pred_class, confidence = predict_image(model, pil_image)
                        
                        # Add prediction text to frame
                        label = f"{pred_class}: {confidence:.2%}"
                        cv2.putText(frame, label, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Display frame
                        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                        
                        # Display prediction
                        prediction_placeholder.markdown(f"""
                        **Current Prediction:** {pred_class}  
                        **Confidence:** {confidence:.2%}
                        """)
                        
                        # Control frame rate
                        time.sleep(0.1)
                        
                        # Check if stop button was pressed
                        if stop_button:
                            st.session_state.webcam_running = False
                            break
                
                except Exception as e:
                    st.error(f"💥 Webcam error: {str(e)}")
                
                finally:
                    cap.release()
                    st.info("📹 Webcam stopped.")
        
        if stop_button:
            st.session_state.webcam_running = False

# --- Additional Information ---
with st.sidebar:
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This app classifies student behavior into:
    - 👀 Attentive
    - 😴 Distracted  
    - 💤 Sleepy
    - 👊 Bullying
    - 💭 Daydreaming
    - 🙋 Hand raising
    - 📱 Phone use
    """)
    
    st.markdown("---")
    st.subheader("System Info")
    st.markdown(f"**Device:** {device}")
    st.markdown(f"**Classes:** {len(classes)}")

if __name__ == "__main__":
    main()