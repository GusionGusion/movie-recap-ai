import streamlit as st
import cv2
import os
import tempfile

st.set_page_config(
    page_title="Movie Recap AI",
    page_icon="🎬"
)

st.title("🎬 Movie Recap AI")
st.write("Upload a movie and analyze video information.")

uploaded_file = st.file_uploader(
    "🎥 Upload Movie / Video",
    type=["mp4", "mov", "avi", "mkv", "webm"]
)

if uploaded_file is not None:

    file_size_mb = uploaded_file.size / (1024 * 1024)

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        st.error("❌ Video file could not be opened.")

    else:

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps > 0:
            duration_seconds = frame_count / fps
        else:
            duration_seconds = 0

        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)

        st.success("✅ Video uploaded successfully!")

        st.subheader("📊 Video Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "File Size",
                f"{file_size_mb:.2f} MB"
            )

            st.metric(
                "Resolution",
                f"{width} × {height}"
            )

        with col2:
            st.metric(
                "Duration",
                f"{minutes} min {seconds} sec"
            )

            st.metric(
                "FPS",
                f"{fps:.2f}"
            )

        st.divider()

        st.subheader("🎥 Video Preview")

        st.video(uploaded_file)

    cap.release()

    try:
        os.remove(video_path)
    except:
        pass
st.divider()

st.subheader("🎤 Movie Transcript")

if st.button("📝 Generate Transcript"):

    st.info("⏳ Transcribing movie audio... Please wait.")

    try:
        import whisper

        model = whisper.load_model("base")

        result = model.transcribe(
            video_path,
            language="en"
        )

        transcript = result["text"]

        st.success("✅ Transcript generated!")

        st.text_area(
            "📄 Transcript",
            transcript,
            height=300
        )

    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")

