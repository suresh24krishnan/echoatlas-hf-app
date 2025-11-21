# agents/speech_agent.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
import av
import numpy as np
import queue

# Optional: TURN/STUN for better connectivity (especially on Hugging Face Spaces)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# A lightweight audio processor that buffers audio frames; transcription is done in chunks.
class AudioBufferProcessor(AudioProcessorBase):
    def __init__(self):
        self.sample_rate = 48000  # WebRTC typical
        self.sample_width = 2     # 16-bit PCM
        self.channels = 1
        self.buffer = queue.Queue()
        self.enabled = True

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to mono 16-bit PCM
        pcm = frame.to_ndarray(layout="mono")
        if pcm is None:
            return frame
        # Normalize dtype to int16 if not already
        if pcm.dtype != np.int16:
            pcm = (pcm * 32767).astype(np.int16) if np.issubdtype(pcm.dtype, np.floating) else pcm.astype(np.int16)

        # Push raw bytes to buffer
        self.buffer.put(pcm.tobytes())
        return frame

# Simple chunk-to-text using Vosk (offline) or SpeechRecognition+Google (optional).
# Here we keep it framework-agnostic and use a tiny placeholder to aggregate bytes;
# you can swap in your preferred STT backend below.
def _bytes_to_text(audio_bytes: bytes, sample_rate: int = 16000):
    # Placeholder: replace with your STT of choice.
    # Options:
    # - Offline: Vosk (recommended for Spaces without external APIs)
    # - Cloud: Azure/Google/OpenAI Whisper (requires API keys)
    #
    # For now, return an empty string to avoid false positives.
    # Integrate real STT below.
    return ""

def get_user_input(location_key: str):
    st.write("🎙️ Click ‘Start’ to enable mic, then speak.")
    webrtc_ctx = webrtc_streamer(
        key="echoatlas-mic",
        mode="recv-only",
        audio_receiver_size=1024,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
    )

    # Live transcript store in session state
    if "echoatlas_transcript" not in st.session_state:
        st.session_state["echoatlas_transcript"] = ""

    # Read buffered audio and attempt lightweight transcription (replace with your STT)
    if webrtc_ctx and webrtc_ctx.audio_receiver:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.01)
        except queue.Empty:
            audio_frames = []

        # Accumulate bytes
        chunk_bytes = b""
        for f in audio_frames:
            # Convert frames to bytes; ensure mono int16
            pcm = f.to_ndarray(layout="mono")
            if pcm is None:
                continue
            if pcm.dtype != np.int16:
                pcm = (pcm * 32767).astype(np.int16) if np.issubdtype(pcm.dtype, np.floating) else pcm.astype(np.int16)
            chunk_bytes += pcm.tobytes()

        # Transcribe chunk
        if len(chunk_bytes) > 0:
            text_chunk = _bytes_to_text(chunk_bytes)
            if text_chunk:
                # Append with a space to form a running transcript
                st.session_state["echoatlas_transcript"] = (
                    f"{st.session_state['echoatlas_transcript']} {text_chunk}".strip()
                )

    # Show transcript
    transcript = st.session_state.get("echoatlas_transcript", "")
    st.write(f"🗣️ Live transcript: {transcript if transcript else 'Listening...'}")

    # Return transcript if we have text; else None
    return transcript if transcript else None
