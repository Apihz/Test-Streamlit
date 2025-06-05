import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# --- Classes ---
classes = ['Attentive', 'Distracted', 'Sleepy', 'Bullying', 'Daydreaming', 'Hand_raising', 'Phone_use']

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model_path = "emotion_model_2.pth"  # Ensure this file exists in the same directory
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# --- Streamlit UI ---
st.title("📸 Real-time Student Attention Detection")
st.markdown("This app uses a webcam feed to classify student behavior in real time.")

frame_placeholder = st.empty()
text_placeholder = st.empty()

# --- Webcam Capture ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Cannot open webcam. Make sure it's connected and not used by another app.")
else:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame from webcam.")
                break

            # Convert frame to RGB for PIL and Torch
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            # Preprocess
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                pred = torch.argmax(probs).item()

            # Label
            label = f"{classes[pred]}: {probs[pred]:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text_placeholder.markdown(f"**Prediction:** {label}")

            # Control frame rate
            time.sleep(0.05)

    except Exception as e:
        st.error(f"💥 Error: {str(e)}")

    finally:
        cap.release()
