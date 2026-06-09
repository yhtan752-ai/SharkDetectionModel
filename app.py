import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Shark Abundance Live Portal", layout="centered")
st.title("Shark Abundance Live Testing Portal")
st.write("Upload an image or a video clip to evaluate the YOLO11 + ByteTrack pipeline smoothly.")

@st.cache_resource
def load_yolo_model():
    return YOLO("my_model.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading model weights file: {e}")
    st.stop()

source_type = st.sidebar.radio("Select Input Source Media Type:", ("Image File", "Video File"))
conf_threshold = st.sidebar.slider("Confidence Threshold:", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

# --- IMAGE HANDLER (STAYS THE SAME) ---
if source_type == "Image File":
    uploaded_image = st.file_uploader("Choose a shark image asset...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="AI Inference Evaluation Output", use_container_width=True)
        st.success(f"Detections complete! Total spotted in image: {len(results[0].boxes)}")

# --- VIDEO HANDLER (SMOOTH BUFFERED PLAYBACK OPTIMIZATION) ---
elif source_type == "Video File":
    uploaded_video = st.file_uploader("Choose a video file tracking target...", type=["mp4", "avi", "mov", "webm"])
    
    if uploaded_video is not None:
        # 1. Setup secure temp paths for input and processing outputs
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        # Build an intermediate tracking draft output path
        output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
        
        cap = cv2.VideoCapture(tfile.name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30
        
        # Use standard MP4V container compiler for the fast backend generation pass
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # 2. Display a nice loading spinner while the AI processes frames in the background
        with st.spinner("🧠 AI Tracking Engine running background inference... Please wait a moment."):
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            maxn_value = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                current_frame += 1
                
                # Run ByteTrack extraction loop 
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_threshold, verbose=False)
                
                sharks_in_this_frame = 0
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    sharks_in_this_frame = len(ids)
                    frame = results[0].plot()
                    
                if sharks_in_this_frame > maxn_value:
                    maxn_value = sharks_in_this_frame
                    
                # Burn telemetry stats onto the processed video tracking frames permanently
                cv2.putText(frame, f'Active Counter: {sharks_in_this_frame}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f'Peak MaxN: {maxn_value}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                out.write(frame)
                
                # Update progress bar metrics securely
                if total_frames > 0:
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
            
            cap.release()
            out.release()
            
        # 3. Stream natively using Streamlit's web player engine container
        st.success(f"🎬 Analysis complete! Overall Peak Video MaxN Score: {maxn_value}")
        
        # Load and play the tracked video file all at once
        with open(output_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
            
        # Clean up backend file memory allocations safely
        try:
            os.remove(tfile.name)
            os.remove(output_path)
        except:
            pass
