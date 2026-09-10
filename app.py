import streamlit as st
import os
from google import genai
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

    st.divider()

st.subheader("🎤 Movie Transcript")

if uploaded_file is not None:

    if st.button("📝 Generate Transcript"):

        st.info("⏳ Transcribing movie audio... Please wait.")

        try:
            import whisper

            model = whisper.load_model("base")

            result = model.transcribe(
                video_path,
                language="en"
            )

            st.session_state["transcript_result"] = result
            st.session_state["transcript"] = result["text"]

            st.success("✅ Transcript generated!")

        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")


# Show saved transcript
if "transcript" in st.session_state:

    st.text_area(
        "📄 Transcript",
        st.session_state["transcript"],
        height=300
    )


st.divider()

st.subheader("🎬 Scene Analysis")

if "transcript_result" in st.session_state:

    if st.button("🎞️ Analyze Scenes"):

        segments = st.session_state["transcript_result"].get(
            "segments",
            []
        )

        if not segments:

            st.warning(
                "⚠️ No timestamped transcript segments found."
            )

        else:

            st.success(
                f"✅ {len(segments)} scenes analyzed!"
            )


    
            for i, segment in enumerate(segments):

                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()

                if text:

                    st.markdown(
                        f"### Scene {i + 1} "
                        f"({start:.1f}s → {end:.1f}s)"
                    )

                    st.write(text)
st.divider()

st.subheader("🤖 AI Scene Analysis")

if "transcript" in st.session_state:

    if st.button("🤖 Analyze with Gemini"):

        try:
            api_key = st.secrets["GEMINI_API_KEY"]

            client = genai.Client(
                api_key=api_key
            )

            transcript_text = st.session_state["transcript"]

            prompt = f"""
You are a professional movie recap analyst.

Analyze the following movie transcript.

For each important scene, provide:

1. Scene number
2. Approximate timestamp if available
3. Characters involved
4. Location
5. Important actions
6. Emotions
7. Short scene summary

Rules:
- Do not invent events.
- Follow the transcript chronology.
- Keep the analysis clear and concise.

Movie Transcript:

{transcript_text}
"""

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            st.session_state["ai_scene_analysis"] = response.text

            st.success("✅ Gemini Scene Analysis completed!")

        except Exception as e:
            st.error(f"❌ Gemini Analysis failed: {e}")


if "ai_scene_analysis" in st.session_state:

    st.text_area(
        "🤖 AI Scene Analysis Result",
        st.session_state["ai_scene_analysis"],
        height=500
    )
    
st.divider()
st.subheader("📝 Movie Recap Script")

if "ai_scene_analysis" in st.session_state:
    if st.button("🎬 Generate Recap Script"):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)

            scene_analysis = st.session_state["ai_scene_analysis"]

            prompt = f"""
You are a professional movie recap script writer.

Using ONLY the scene analysis below, write a complete
movie recap narration script.

Rules:
- Follow chronological order.
- Do not invent events.
- Focus on important story events.
- Remove unnecessary repetition.
- Make the narration engaging and easy to understand.
- Write naturally for voiceover.
- Write only the recap narration.

Scene Analysis:

{scene_analysis}
"""

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            st.session_state["recap_script"] = response.text
            st.success("✅ Movie Recap Script generated!")

        except Exception as e:
            st.error(f"❌ Recap generation failed: {e}")

if "recap_script" in st.session_state:
    st.text_area(
        "🎬 Recap Script",
        st.session_state["recap_script"],
        height=600
    )
