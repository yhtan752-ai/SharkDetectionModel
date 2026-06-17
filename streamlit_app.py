import streamlit as st
import requests

st.title("High-Speed Shark Tracking System (GPU Accelerated)")

# Update this to your active HTTPS Pinggy link
# Note the endpoint changes back to /process-video/
GPU_SERVER_URL = "http://10.3.250.183:8000/process-video/"
uploaded_file = st.file_uploader("Upload Shark Footage", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    if st.button("Run GPU Tracking Pipeline"):
        with st.spinner("🚀 Server GPU is processing your video... Please wait."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Give the GPU plenty of processing overhead time
                response = requests.post(GPU_SERVER_URL, files=files, timeout=(30, 300))
                
                if response.status_code == 200:
                    st.success("GPU Processing Complete!")
                    # Display the final processed video payload directly on the page!
                    st.video(response.content)
                else:
                    st.error(f"Server Error: Received Status Code {response.status_code}")
                    
            except Exception as e:
                st.error(f"Network Connection Failed: {str(e)}")
