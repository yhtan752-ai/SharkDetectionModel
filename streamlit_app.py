import streamlit as st
import requests

# Point this to your lecturer's GPU server public IP address
# Inside your local streamlit_app.py or your deployed file:
GPU_SERVER_URL = "https://thick-sides-bow.loca.lt/process-video/"

st.title("🦈 Automated Real-Time Shark Detection & Counter")
st.write("Upload a video stream below. Processing is offloaded to a high-performance external GPU cluster.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file) # Preview original file
    
    if st.button("Run Analytics Engine"):
        with st.spinner("Offloading video tensor batches to GPU server... Please wait."):
            try:
                # Prepare file payload payload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send HTTP POST request containing video bytes across network to server
                response = requests.post(GPU_SERVER_URL, files=files, timeout=(15, 300))
                
                if response.status_code == 200:
                    st.success("GPU Processing Complete!")
                    # Display the final returned tracking video seamlessly
                    st.video(response.content)
                else:
                    st.error(f"Server Error: Received Status Code {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.error("The processing request timed out. The video file might be too large.")
            except requests.exceptions.ConnectionError:
                st.error("Could not establish a connection to the remote GPU Server. Ensure the server API is active.")
