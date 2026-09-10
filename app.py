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

    st.session_state["uploaded_file"] = uploaded_file.getvalue()

    file_size_mb = uploaded_file.size / (1024 * 1024)

    video_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp_file:

        temp_file.write(video_bytes)
        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps else 0

    cap.release()

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
st.divider()
st.subheader("🇲🇲 Myanmar Recap Script")

if "recap_script" in st.session_state:

    if st.button("🇲🇲 Translate to Myanmar"):

        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)

            recap_script = st.session_state["recap_script"]

            prompt = f"""
Translate the following movie recap narration into natural
Myanmar Burmese language.

Rules:
- Preserve the original meaning exactly.
- Do not add or remove story events.
- Use natural spoken Myanmar Burmese.
- Make it suitable for female voiceover.
- Keep chronological order.
- Do not include English.
- Write only the Myanmar narration.

English Recap:

{recap_script}
"""

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            st.session_state["myanmar_recap"] = response.text

            st.success("✅ Myanmar Recap Script generated!")

        except Exception as e:
            st.error(f"❌ Myanmar translation failed: {e}")


if "myanmar_recap" in st.session_state:

    st.text_area(
        "🇲🇲 Myanmar Recap",
        st.session_state["myanmar_recap"]
    )


st.divider()
st.subheader("🎙️ Myanmar Female Voiceover")

if "myanmar_recap" in st.session_state:

    if st.button("🎙️ Generate Myanmar Voiceover"):

        try:
            import edge_tts
            import asyncio

            text = st.session_state["myanmar_recap"]

            async def create_voice():
                communicate = edge_tts.Communicate(
                    text,
                    "my-MM-NilarNeural"
                )
                await communicate.save("myanmar_voiceover.mp3")

            asyncio.run(create_voice())

            st.session_state["voiceover_file"] = "myanmar_voiceover.mp3"

            st.success("✅ Myanmar Female Voiceover generated!")

        except Exception as e:
            st.error(f"❌ Voiceover generation failed: {e}")


if "voiceover_file" in st.session_state:

    st.audio(
        st.session_state["voiceover_file"],
        format="audio/mp3"
    )
st.divider()
st.subheader("🇲🇲 Myanmar Subtitle")

if "myanmar_recap" in st.session_state:

    if st.button("💬 Generate Myanmar Subtitle"):

        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)

            myanmar_text = st.session_state["myanmar_recap"]

            prompt = f"""
Create Myanmar subtitles for the following movie recap narration.

Rules:
- Write only Myanmar Burmese.
- Split the narration into short subtitle segments.
- Each segment should be easy to read.
- Maximum 12 words per subtitle.
- Keep the original meaning.
- Do not invent information.
- Return ONLY valid JSON.
- Do not use markdown.

Required JSON format:

[
  {{
    "start": 0,
    "end": 5,
    "text": "မြန်မာစာတန်းထိုး"
  }},
  {{
    "start": 5,
    "end": 10,
    "text": "နောက်ထပ် မြန်မာစာတန်းထိုး"
  }}
]

Myanmar Recap:

{myanmar_text}
"""

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            import json

            subtitle_data = json.loads(response.text)

            st.session_state["subtitle_data"] = subtitle_data

            st.success("✅ Myanmar Subtitle generated!")

        except Exception as e:
            st.error(f"❌ Subtitle generation failed: {e}")


if "subtitle_data" in st.session_state:

    st.text_area(
        "💬 Myanmar Subtitle Preview",
        "\n".join(
            [
                f'{item["start"]:.1f}s → {item["end"]:.1f}s | {item["text"]}'
                for item in st.session_state["subtitle_data"]
            ]
        ),
        height=400
    )
st.subheader("📥 Subtitle Export")

if "subtitle_data" in st.session_state:

    srt_content = ""

    for i, item in enumerate(
        st.session_state["subtitle_data"],
        start=1
    ):

        start = float(item["start"])
        end = float(item["end"])
        text = item["text"].strip()

        start_h = int(start // 3600)
        start_m = int((start % 3600) // 60)
        start_s = int(start % 60)
        start_ms = int((start % 1) * 1000)

        end_h = int(end // 3600)
        end_m = int((end % 3600) // 60)
        end_s = int(end % 60)
        end_ms = int((end % 1) * 1000)

        start_time = (
            f"{start_h:02d}:{start_m:02d}:"
            f"{start_s:02d},{start_ms:03d}"
        )

        end_time = (
            f"{end_h:02d}:{end_m:02d}:"
            f"{end_s:02d},{end_ms:03d}"
        )

        srt_content += (
            f"{i}\n"
            f"{start_time} --> {end_time}\n"
            f"{text}\n\n"
        )

    st.download_button(
        "📥 Download Myanmar Subtitle (.srt)",
        data=srt_content.encode("utf-8"),
        file_name="myanmar_subtitles.srt",
        mime="text/plain"
    )
st.divider()
st.subheader("🎬 Final Video Export")

if (
    "voiceover_file" in st.session_state
    and "subtitle_data" in st.session_state
    and "uploaded_file" in st.session_state
):

    if st.button("🎬 Create Final Recap Video"):

        try:
            import subprocess

            video_file = st.session_state["uploaded_file"]

            output_video = "final_movie_recap.mp4"
            subtitle_file = "myanmar_subtitles.srt"

            # Save uploaded video
            with open("original_movie.mp4", "wb") as f:
                f.write(video_file)

            # Create SRT file
            srt_content = ""

            for i, item in enumerate(
                st.session_state["subtitle_data"],
                start=1
            ):

                start = float(item["start"])
                end = float(item["end"])
                text = item["text"].strip()

                def srt_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 1000)

                    return (
                        f"{hours:02d}:{minutes:02d}:"
                        f"{secs:02d},{millis:03d}"
                    )

                srt_content += (
                    f"{i}\n"
                    f"{srt_time(start)} --> {srt_time(end)}\n"
                    f"{text}\n\n"
                )

            with open(
                subtitle_file,
                "w",
                encoding="utf-8"
            ) as f:
                f.write(srt_content)

            # Add Myanmar voiceover + burn Myanmar subtitles

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        "original_movie.mp4",
        "-i",
        st.session_state["voiceover_file"],
        "-vf",
        "subtitles=myanmar_subtitles.srt",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-shortest",
        output_video
    ],
    check=True
)
            )

            st.success("✅ Final video created!")

            with open(
                output_video,
                "rb"
            ) as f:

                st.download_button(
                    "📥 Download Final Recap Video",
                    data=f,
                    file_name="final_movie_recap.mp4",
                    mime="video/mp4"
                )

        except Exception as e:

            st.error(
                f"❌ Final video export failed: {e}"
            )
