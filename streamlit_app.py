import streamlit as st
import requests
import cv2
import tempfile
import numpy as np

st.title("Shark Tracking System")

# 1. Update this to your active HTTPS Pinggy link
# Note the endpoint changed from /process-video/ to /process-frame/
FRAME_API_URL = "https://qkrez-2406-3003-2000-7027-10bb-b113-617a-a5cb.run.pinggy-free.link/process-frame/"

uploaded_file = st.file_uploader("Upload Shark Footage", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.success("Video received successfully! Commencing streaming frame tracking...")
    
    # Save uploaded file bytes to a temporary local cache file to read via OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Create an empty placeholder container on the Streamlit screen to update the frames live
    frame_placeholder = st.empty()
    
    # Process and stream the video frame-by-frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Encode individual frame array into JPEG bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        try:
            # Send the tiny, lightweight frame down the tunnel
            response = requests.post(
                FRAME_API_URL, 
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")}, 
                timeout=5
            )
            
            if response.status_code == 200:
                # Reconstruct the annotated tracking image returned by the server
                nparr = np.frombuffer(response.content, np.uint8)
                annotated_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # Convert BGR back to standard RGB for Streamlit rendering
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update the display layout with the tracked bounding boxes live!
                frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
            else:
                st.error(f"Backend frame syncing issue: Status {response.status_code}")
                break
                
        except Exception as e:
            st.error(f"Network processing interruption: {str(e)}")
            break
            
    cap.release()
    st.balloons()
