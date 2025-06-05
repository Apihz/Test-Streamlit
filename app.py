import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

# Your classes
classes = ['Attentive', 'Distracted', 'Sleepy', 'Bullying', 'Daydreaming', 'Hand_raising', 'Phone_use']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = "emotion_model_2.pth"
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

st.title("Real-time Attention Detection")

# Placeholder to show video frames
frame_placeholder = st.empty()
text_placeholder = st.empty()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Cannot open webcam")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break
        
        # Convert BGR(OpenCV) to RGB(PIL)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        
        # Preprocess and predict
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred = torch.argmax(probs).item()
        
        # Draw prediction text on frame
        label = f"{classes[pred]}: {probs[pred]:.2f}"
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,0), 2, cv2.LINE_AA)
        
        # Show frame with prediction
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text_placeholder.text(label)
        
        # To control frame rate, add a short delay
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
