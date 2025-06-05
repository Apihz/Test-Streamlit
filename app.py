import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import av

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Classes
classes = ['Attentive', 'Distracted', 'Sleepy', 'Bullying', 'Daydreaming', 'Hand_raising', 'Phone_use']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model_path = "emotion_model_2.pth"
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

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
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to PIL Image for your transform & model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Preprocess & predict
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred = torch.argmax(probs).item()
            conf = probs[pred].item()

        label = f"{classes[pred]}: {conf:.2%}"

        # Put label text on frame
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("📸 Real-time Student Attention Detection")
    st.write("Webcam feed with live attention analysis.")

    if model is None:
        st.error("Model not loaded. Please upload your model file.")
        return

    webrtc_streamer(key="attention-detection", video_transformer_factory=AttentionTransformer)

if __name__ == "__main__":
    main()
