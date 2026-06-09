import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Set up page styling
st.set_page_config(page_title="Shark MaxN Live Portal", layout="centered")
st.title("Shark MaxN Live Portal")
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
        
        # Render boundary boxes natively on image array copy
        annotated_frame = results[0].plot()
        
        # Display the live prediction snapshot to the user interface
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="AI Inference Evaluation Output", use_container_width=True)
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
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                sharks_in_this_frame = len(ids)
                
                # Draw boxes natively
                frame = results[0].plot()
                
            if sharks_in_this_frame > maxn_value:
                maxn_value = sharks_in_this_frame
                
            # Overlay UI text tracks onto array output frame block before display
            cv2.putText(frame, f'Active Counter: {sharks_in_this_frame}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f'Peak MaxN: {maxn_value}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Keep sending the fresh processed frame array right to your web layout
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
        cap.release()
        os.remove(tfile.name) # Cleanup temp file space allocations safely
        st.success(f"Video Processing Complete! Video Peak MaxN Abundance Score: {maxn_value}")