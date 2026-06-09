import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Set up page styling
st.set_page_config(page_title="Shark Abundance Live Portal", layout="centered")
st.title("🦈 Shark Abundance Live Testing Portal")
st.write("Upload an image or a video clip to evaluate the YOLO11 + ByteTrack pipeline live.")

# 1. Load Model Weights securely
@st.cache_resource
def load_yolo_model():
    # Caches the model in server memory so it doesn't reload on every click
    return YOLO("my_model.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading model weights file: {e}")
    st.stop()

# 2. Source Selector Sidebar UI
source_type = st.sidebar.radio("Select Input Source Media Type:", ("Image File", "Video File"))
conf_threshold = st.sidebar.slider("Confidence Threshold:", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

# 3. Handle Live IMAGE Testing Pipeline
if source_type == "Image File":
    uploaded_image = st.file_uploader("Choose a shark image asset...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Convert uploaded bytes to OpenCV matrix format
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Run inference frame snapshot
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Manually parse boxes to display custom names for images
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                
                # FIXED: Updated string structure exactly as requested
                custom_label = f"Blacktip Reef Shark Confidence Score : {int(conf * 100)}%"
                
                (text_w, text_h), _ = cv2.getTextSize(custom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (box[0], box[1] - text_h - 10), (box[0] + text_w + 10, box[1]), (255, 0, 0), -1)
                cv2.putText(frame, custom_label, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display the live prediction snapshot to the user interface
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="AI Inference Evaluation Output", use_container_width=True)
        st.success(f"Detections complete! Total spotted in image: {len(results[0].boxes)}")

# 4. Handle Live VIDEO Testing Pipeline
elif source_type == "Video File":
    uploaded_video = st.file_uploader("Choose a video file tracking target...", type=["mp4", "avi", "mov", "webm"])
    
    if uploaded_video is not None:
        # Streamlit requires parsing video streams via temporary filesystem files
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        
        st.info("AI Analysis Engine is unpacking frames... Streaming live pipeline processing placeholder below.")
        
        # Set up a real-time frame placeholder box in the web window
        frame_placeholder = st.empty()
        
        # Track situational metrics
        maxn_value = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Run tracker channel matching loop block
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_threshold, verbose=False)
            
            sharks_in_this_frame = 0
            if results[0].boxes.id is not None:
                # Extracted metrics directly from the model object tensors
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                sharks_in_this_frame = len(boxes)
                
                # Loop through each detection to override the label strings manually
                for box, conf in zip(boxes, confidences):
                    # Draw neon bounding box frame
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    
                    # FIXED: Updated string structure exactly as requested
                    custom_label = f"Blacktip Reef Shark Confidence Score : {int(conf * 100)}%"
                    
                    # Draw solid text background banner (Slightly adjusted text scale so the longer label fits perfectly)
                    (text_w, text_h), _ = cv2.getTextSize(custom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (box[0], box[1] - text_h - 10), (box[0] + text_w + 10, box[1]), (255, 0, 0), -1)
                    
                    # Print custom white metadata text onto the solid block background
                    cv2.putText(frame, custom_label, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
            if sharks_in_this_frame > maxn_value:
                maxn_value = sharks_in_this_frame
                
            # BOLD TELEMETRY: Large, bold counters offset safely from the screen margins
            cv2.putText(frame, f'Active Counter: {sharks_in_this_frame}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(frame, f'Peak MaxN: {maxn_value}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Keep sending the fresh processed frame array right to your web layout
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
        cap.release()
        os.remove(tfile.name) # Cleanup temp file space allocations safely
        st.success(f"Video Processing Complete! Video Peak MaxN Abundance Score: {maxn_value}")
