# MUST BE FIRST - Set page config before any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Student Attention Detection",
    page_icon="📸",
    layout="wide"
)

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import av
import asyncio
import sys
import threading

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Fix for event loop issues
def fix_event_loop():
    """Fix asyncio event loop issues in Streamlit"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, create one
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Apply the fix
fix_event_loop()

# Classes
classes = ['Attentive', 'Distracted', 'Sleepy', 'Bullying', 'Daydreaming', 'Hand_raising', 'Phone_use']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with better error handling
@st.cache_resource
def load_model():
    """Load the attention detection model"""
    model_path = "emotion_model_2.pth"
    
    try:
        # Create model architecture
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        return model
        
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Video Transformer for real-time prediction
class AttentionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.transform = transform
        self.device = device
        self.classes = classes
        
    def recv(self, frame):
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # If model is not loaded, just return original frame
            if self.model is None:
                cv2.putText(img, "Model not loaded", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Convert BGR to RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess image
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                pred = torch.argmax(probs).item()
                conf = probs[pred].item()
            
            # Create label
            label = f"{self.classes[pred]}: {conf:.2%}"
            
            # Choose color based on attention state
            color = (0, 255, 0) if self.classes[pred] == 'Attentive' else (0, 165, 255)
            if self.classes[pred] in ['Sleepy', 'Distracted']:
                color = (0, 0, 255)
            
            # Add label to frame
            cv2.putText(img, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add confidence bar
            bar_width = int(300 * conf)
            cv2.rectangle(img, (10, 50), (10 + bar_width, 70), color, -1)
            cv2.rectangle(img, (10, 50), (310, 70), (255, 255, 255), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # If any error occurs, return original frame with error message
            cv2.putText(img, f"Error: {str(e)[:30]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("📸 Real-time Student Attention Detection")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.write("This app detects student attention states in real-time using your webcam.")
        
        st.subheader("Detected States:")
        for i, class_name in enumerate(classes):
            emoji = "✅" if class_name == "Attentive" else "⚠️" if class_name in ["Hand_raising"] else "❌"
            st.write(f"{emoji} {class_name}")
        
        st.subheader("Requirements:")
        st.write("- Webcam access")
        st.write("- Good lighting")
        st.write("- Model file: emotion_model_2.pth")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        if model is None:
            st.error("⚠️ Model not loaded! Please ensure 'emotion_model_2.pth' is in the app directory.")
            st.info("The camera will still work, but predictions won't be available.")
        else:
            st.success("✅ Model loaded and ready!")
        
        # WebRTC streamer with better configuration
        ctx = webrtc_streamer(
            key="attention-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=AttentionTransformer,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"min": 15, "ideal": 30, "max": 60}
                },
                "audio": False
            },
            async_processing=True,
        )
    
    with col2:
        st.subheader("📊 System Status")
        
        # Device info
        st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name()}")
        
        # Model info
        if model is not None:
            st.metric("Model Status", "✅ Loaded")
            st.metric("Classes", len(classes))
        else:
            st.metric("Model Status", "❌ Not Loaded")
        
        # Instructions
        st.subheader("📋 Instructions")
        st.write("""
        1. Click 'START' to begin camera feed
        2. Allow camera permissions
        3. Position yourself in the camera view
        4. Watch real-time attention detection
        5. Click 'STOP' to end session
        """)

if __name__ == "__main__":
    main()