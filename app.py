import streamlit as st
import os
from google import genai
from google.genai import types
import cv2
import tempfile
import subprocess
import math
import asyncio
import json
import re
import shutil
import base64
import streamlit.components.v1 as components


# =========================================================
# LOCAL INTERACTIVE BLUR VIDEO COMPONENT
# =========================================================

BLUR_VIDEO_EDITOR = components.declare_component(
    "blur_video_editor",
    path=os.path.join(
        os.path.dirname(__file__),
        "blur_component"
    )
)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Movie Recap AI",
    page_icon="🎬"
)

st.title("🎬 Movie Recap AI")
st.write("Upload a movie and analyze video information.")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_audio_duration(media_file):
    """Read exact media duration using ffprobe."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                media_file
            ],
            capture_output=True,
            text=True,
            check=True
        )

        value = result.stdout.strip()

        if not value:
            return 0.0

        return float(value)

    except Exception:
        return 0.0


def extract_audio_for_whisper(video_path):
    """
    Extract clean mono 16 kHz WAV for Faster-Whisper.
    """

    audio_file = os.path.join(
        tempfile.gettempdir(),
        f"movie_recap_whisper_{os.getpid()}.wav"
    )

    if os.path.exists(audio_file):
        try:
            os.remove(audio_file)
        except Exception:
            pass

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            audio_file
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg audio extraction failed:\n"
            + result.stderr[-3000:]
        )

    if not os.path.exists(audio_file):
        raise RuntimeError(
            "Audio file was not created."
        )

    file_size = os.path.getsize(
        audio_file
    )

    if file_size < 1000:
        raise RuntimeError(
            "Extracted audio file is too small."
        )

    duration = get_audio_duration(
        audio_file
    )

    if duration <= 0:
        raise RuntimeError(
            "Extracted audio duration is invalid."
        )

    return audio_file


def ass_time(seconds):
    """Convert seconds to ASS timestamp."""

    seconds = max(
        0,
        float(seconds)
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    centiseconds = int(
        (seconds % 1) * 100
    )

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{centiseconds:02d}"
    )


def split_myanmar_text(
    text,
    max_chars=65
):
    """
    Split Myanmar narration into TTS/subtitle chunks.
    Sentence boundaries are preferred.
    """

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if not text:
        return []

    sentences = re.split(
        r"(?<=[။!?])\s+|(?<=[.!?])\s+",
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    chunks = []

    for sentence in sentences:

        if len(sentence) <= max_chars:

            chunks.append(
                sentence
            )

            continue

        words = sentence.split()

        current = ""

        for word in words:

            candidate = (
                word
                if not current
                else current + " " + word
            )

            if len(candidate) <= max_chars:

                current = candidate

            else:

                if current:

                    chunks.append(
                        current
                    )

                current = word

        if current:

            chunks.append(
                current
            )

    return chunks


def wrap_myanmar(
    text,
    max_chars=24
):
    """Wrap subtitle into a maximum of 2 ASS lines without dropping text."""

    text = re.sub(
        r"\s+",
        " ",
        str(text).strip()
    )

    if not text:
        return ""

    words = text.split()

    lines = []

    current = ""

    for word in words:

        candidate = (
            word
            if not current
            else current + " " + word
        )

        if (
            len(candidate) <= max_chars
            or not current
        ):

            current = candidate

        elif len(lines) == 0:

            lines.append(
                current
            )

            current = word

        else:

            current += " " + word

    if current:
        lines.append(
            current
        )

    if len(lines) > 2:

        lines = [
            lines[0],
            " ".join(lines[1:])
        ]

    return "\\N".join(
        lines[:2]
    )


# =========================================================
# VIDEO FRAME HELPER
# =========================================================

def extract_frame_bytes(
    cap,
    timestamp
):
    """
    Extract one actual frame from video.
    """

    try:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            max(
                0,
                float(timestamp)
            ) * 1000
        )

        ret, frame = cap.read()

        if not ret or frame is None:

            return None

        max_side = 768

        h, w = frame.shape[:2]

        scale = min(
            1.0,
            max_side / max(
                h,
                w
            )
        )

        if scale < 1.0:

            new_w = int(
                w * scale
            )

            new_h = int(
                h * scale
            )

            frame = cv2.resize(
                frame,
                (
                    new_w,
                    new_h
                ),
                interpolation=cv2.INTER_AREA
            )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                82
            ]
        )

        if not success:

            return None

        return encoded.tobytes()

    except Exception:

        return None


# =========================================================
# CACHED FASTER-WHISPER MODEL
# =========================================================

@st.cache_resource
def load_whisper_model(
    model_name="tiny"
):

    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1
    )


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "🎥 Upload Movie / Video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm"
    ]
)


if uploaded_file is not None:

    video_bytes = uploaded_file.getvalue()

    st.session_state[
        "uploaded_file"
    ] = video_bytes

    file_size_mb = (
        uploaded_file.size
        / (1024 * 1024)
    )

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            video_bytes
        )

        video_path = temp_file.name

    st.session_state[
        "video_path"
    ] = video_path

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        st.error(
            "❌ Video file could not be opened."
        )

    else:

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if fps > 0:

            duration_seconds = (
                frame_count / fps
            )

        else:

            duration_seconds = 0

        minutes = int(
            duration_seconds // 60
        )

        seconds = int(
            duration_seconds % 60
        )

        st.success(
            "✅ Video uploaded successfully!"
        )

        st.subheader(
            "📊 Video Information"
        )

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

        st.session_state[
            "video_duration"
        ] = duration_seconds

        st.session_state[
            "video_width"
        ] = width

        st.session_state[
            "video_height"
        ] = height

        st.session_state[
            "video_fps"
        ] = fps

        st.divider()

        st.subheader(
            "🎥 Video Preview"
        )

        st.video(
            video_bytes
        )

        cap.release()

    st.divider()


# =========================================================
# ONE CLICK RECAP SETTINGS
# =========================================================

st.subheader(
    "⚙️ Recap Settings"
)

settings_col1, settings_col2 = st.columns(2)


with settings_col1:

    whisper_model = st.selectbox(
        "🧠 Whisper Model",
        [
            "tiny",
            "base"
        ],
        index=0,
        help=(
            "Tiny uses much less CPU/RAM. "
            "Base may provide better English transcription."
        ),
        key="main_whisper_model"
    )

    voice_gender = st.selectbox(
        "🎙️ Voice",
        [
            "👩 Female — Nilar",
            "👨 Male — Thiha"
        ],
        key="main_voice_gender"
    )

    voice_speed = st.selectbox(
        "🎚️ Voice Speed",
        [
            1.0,
            1.1,
            1.2
        ],
        index=0,
        key="main_voice_speed"
    )


with settings_col2:

    freeze_enabled = st.checkbox(
        "🧊 Enable Freeze Frame + Zoom",
        value=True,
        key="main_freeze_enabled"
    )

    freeze_interval = st.number_input(
        "⏱️ Freeze Every",
        min_value=5.0,
        max_value=60.0,
        value=10.0,
        step=1.0,
        key="main_freeze_interval"
    )

    freeze_duration = st.number_input(
        "🧊 Freeze Duration",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        key="main_freeze_duration"
    )

    aspect_ratio = st.selectbox(
        "📐 Output Aspect Ratio",
        [
            "Original",
            "9:16",
            "16:9",
            "1:1",
            "3:4"
        ],
        index=0,
        key="main_aspect_ratio"
    )

    subtitle_font = st.selectbox(
        "🔤 Subtitle Font",
        [
            "Noto Sans Myanmar",
            "Pyidaungsu",
            "Myanmar Text",
            "Custom Font"
        ],
        index=0,
        key="main_subtitle_font"
    )

    custom_font_file = st.file_uploader(
        "📁 Upload Custom Font (.ttf/.otf)",
        type=[
            "ttf",
            "otf"
        ],
        key="main_custom_font"
    )

    subtitle_size = st.slider(
        "🔠 Subtitle Font Size",
        min_value=20,
        max_value=100,
        value=50,
        step=1,
        key="main_subtitle_size"
    )


# =========================================================
# ORIGINAL SUBTITLE BLUR TOOL
# =========================================================

st.subheader(
    "🔲 Original Subtitle Blur"
)

blur_enabled = st.checkbox(
    "🔲 Blur Original Movie Subtitle",
    value=False,
    key="main_blur_enabled"
)


# =========================================================
# INITIAL BLUR STATE
# =========================================================

if uploaded_file is not None:

    blur_source_key = (
        uploaded_file.name
        + "_"
        + str(uploaded_file.size)
    )

    if (
        st.session_state.get(
            "_blur_source_key"
        )
        != blur_source_key
    ):

        st.session_state[
            "_blur_source_key"
        ] = blur_source_key

        st.session_state[
            "main_blur_x_value"
        ] = 10.0

        st.session_state[
            "main_blur_y_value"
        ] = 78.0

        st.session_state[
            "main_blur_width_value"
        ] = 80.0

        st.session_state[
            "main_blur_height_value"
        ] = 18.0


# =========================================================
# INTERACTIVE BLUR VIDEO PREVIEW
# =========================================================

if blur_enabled:

    st.markdown(
        "### 🎯 Blur Box Preview"
    )

    st.caption(
        "🎥 ဒီနေရာမှာ Uploaded Video ကို တကယ်ကြည့်နိုင်ပါတယ်။ "
        "📱 Mobile မှာ Blur Box အလယ်ကို လက်နဲ့ဆွဲပြီး Move လုပ်ပါ။ "
        "Corner / Edge ကို ဆွဲပြီး Resize လုပ်ပါ။"
    )

    if (
        "uploaded_file" not in st.session_state
        or
        "video_path" not in st.session_state
        or not os.path.exists(
            st.session_state["video_path"]
        )
    ):

        st.warning(
            "⚠️ Blur Preview အတွက် video အရင် upload လုပ်ပါ။"
        )

    else:

        # -------------------------------------------------
        # Actual uploaded video -> Base64 Data URL
        # -------------------------------------------------

        video_bytes_for_preview = (
            st.session_state[
                "uploaded_file"
            ]
        )

        mime_type = (
            uploaded_file.type
            if uploaded_file is not None
            and uploaded_file.type
            else "video/mp4"
        )

        video_base64 = base64.b64encode(
            video_bytes_for_preview
        ).decode(
            "ascii"
        )

        video_data_url = (
            f"data:{mime_type};base64,"
            f"{video_base64}"
        )

        # -------------------------------------------------
        # Current Blur Values
        # -------------------------------------------------

        current_blur_x = float(
            st.session_state.get(
                "main_blur_x_value",
                10.0
            )
        )

        current_blur_y = float(
            st.session_state.get(
                "main_blur_y_value",
                78.0
            )
        )

        current_blur_width = float(
            st.session_state.get(
                "main_blur_width_value",
                80.0
            )
        )

        current_blur_height = float(
            st.session_state.get(
                "main_blur_height_value",
                18.0
            )
        )

        # -------------------------------------------------
        # Interactive Video Editor
        # -------------------------------------------------

        blur_result = BLUR_VIDEO_EDITOR(
            video_src=video_data_url,
            initial_x=current_blur_x,
            initial_y=current_blur_y,
            initial_width=current_blur_width,
            initial_height=current_blur_height,
            key="blur_video_editor"
        )

        # -------------------------------------------------
        # Receive Box Position from JavaScript
        # -------------------------------------------------

        if isinstance(
            blur_result,
            dict
        ):

            try:

                new_x = float(
                    blur_result.get(
                        "x",
                        current_blur_x
                    )
                )

                new_y = float(
                    blur_result.get(
                        "y",
                        current_blur_y
                    )
                )

                new_width = float(
                    blur_result.get(
                        "width",
                        current_blur_width
                    )
                )

                new_height = float(
                    blur_result.get(
                        "height",
                        current_blur_height
                    )
                )

                # -------------------------------------------------
                # Safety Clamp
                # -------------------------------------------------

                new_width = max(
                    1.0,
                    min(
                        100.0,
                        new_width
                    )
                )

                new_height = max(
                    1.0,
                    min(
                        100.0,
                        new_height
                    )
                )

                new_x = max(
                    0.0,
                    min(
                        100.0 - new_width,
                        new_x
                    )
                )

                new_y = max(
                    0.0,
                    min(
                        100.0 - new_height,
                        new_y
                    )
                )

                st.session_state[
                    "main_blur_x_value"
                ] = new_x

                st.session_state[
                    "main_blur_y_value"
                ] = new_y

                st.session_state[
                    "main_blur_width_value"
                ] = new_width

                st.session_state[
                    "main_blur_height_value"
                ] = new_height

            except Exception:

                pass


    # ---------------------------------------------------------
    # Blur Strength
    # ---------------------------------------------------------

    blur_strength = st.slider(
        "💪 Blur Strength",
        min_value=1,
        max_value=30,
        value=12,
        step=1,
        key="main_blur_strength"
    )


    # ---------------------------------------------------------
    # Use Interactive Values
    # ---------------------------------------------------------

    blur_x = float(
        st.session_state.get(
            "main_blur_x_value",
            10.0
        )
    )

    blur_y = float(
        st.session_state.get(
            "main_blur_y_value",
            78.0
        )
    )

    blur_width = float(
        st.session_state.get(
            "main_blur_width_value",
            80.0
        )
    )

    blur_height = float(
        st.session_state.get(
            "main_blur_height_value",
            18.0
        )
    )


    st.success(
        "✅ Blur Box position saved."
    )

    st.caption(
        f"X: {blur_x:.1f}%  |  "
        f"Y: {blur_y:.1f}%  |  "
        f"W: {blur_width:.1f}%  |  "
        f"H: {blur_height:.1f}%"
    )


    st.info(
        f"🔲 Selected Blur Area: "
        f"X {blur_x:.1f}% | "
        f"Y {blur_y:.1f}% | "
        f"W {blur_width:.1f}% | "
        f"H {blur_height:.1f}% | "
        f"Strength {blur_strength}"
    )

else:

    st.caption(
        "Original subtitle blur is OFF."
    )


st.caption(
    "⚙️ Set your preferred settings first, "
    "then press One Click."
)

st.divider()


# =========================================================
# ONE CLICK BUTTON
# =========================================================

run_all = False

if uploaded_file is not None:

    st.markdown(
        "### 🎬 Ready to Generate"
    )

    run_all = st.button(
        "🎬 ONE CLICK — GENERATE MOVIE RECAP",
        type="primary",
        use_container_width=True
    )

    if run_all:

        st.success(
            "🚀 One Click Recap started!"
        )

st.divider()


# =========================================================
# MOVIE TRANSCRIPT
# =========================================================

st.subheader(
    "🎤 Movie Transcript"
)


if uploaded_file is not None:

    if st.button(
        "📝 Generate Transcript"
    ) or run_all:

        st.info(
            f"⏳ Transcribing with Faster-Whisper "
            f"{whisper_model}..."
        )

        try:

            audio_path = extract_audio_for_whisper(
                st.session_state[
                    "video_path"
                ]
            )

            audio_duration = get_audio_duration(
                audio_path
            )

            st.write(
                f"🔊 Audio Duration: "
                f"{audio_duration:.2f} seconds"
            )

            model = load_whisper_model(
                whisper_model
            )

            segments_generator, info = model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500
                }
            )

            transcript_segments = []

            for segment in segments_generator:

                text = str(
                    segment.text
                ).strip()

                if not text:
                    continue

                transcript_segments.append(
                    {
                        "start": float(
                            segment.start
                        ),
                        "end": float(
                            segment.end
                        ),
                        "text": text
                    }
                )

            if not transcript_segments:

                raise RuntimeError(
                    "Faster-Whisper returned no transcript."
                )

            transcript_text = "\n".join(
                [
                    (
                        f"[{seg['start']:.2f} - "
                        f"{seg['end']:.2f}] "
                        f"{seg['text']}"
                    )
                    for seg in transcript_segments
                ]
            )

            result = {
                "text": transcript_text,
                "segments": transcript_segments
            }

            st.session_state[
                "transcript_result"
            ] = result

            st.session_state[
                "transcript"
            ] = transcript_text

            st.success(
                f"✅ Transcript generated! "
                f"{len(transcript_segments)} segments"
            )

        except Exception as e:

            st.error(
                f"❌ Transcription failed: {e}"
            )


if "transcript" in st.session_state:

    st.text_area(
        "📄 Transcript",
        st.session_state[
            "transcript"
        ],
        height=300
    )


st.divider()


# =========================================================
# SCENE ANALYSIS
# =========================================================

st.subheader(
    "🎬 Scene Analysis"
)


if "transcript_result" in st.session_state:

    if st.button(
        "🎞️ Analyze Scenes"
    ) or run_all:

        segments = (
            st.session_state[
                "transcript_result"
            ].get(
                "segments",
                []
            )
        )

        if not segments:

            st.warning(
                "⚠️ No timestamped transcript segments found."
            )

        else:

            try:

                api_key = st.secrets[
                    "GEMINI_API_KEY"
                ]

                client = genai.Client(
                    api_key=api_key
                )

                video_path = (
                    st.session_state[
                        "video_path"
                    ]
                )

                cap = cv2.VideoCapture(
                    video_path
                )

                if not cap.isOpened():

                    st.error(
                        "❌ Could not open video for scene analysis."
                    )

                else:

                    scene_items = []

                    usable_segments = [
                        seg
                        for seg in segments
                        if str(
                            seg.get(
                                "text",
                                ""
                            )
                        ).strip()
                    ]

                    max_frames = 24

                    if len(
                        usable_segments
                    ) > max_frames:

                        selected_indexes = []

                        for n in range(
                            max_frames
                        ):

                            pos = (
                                n
                                * (
                                    len(
                                        usable_segments
                                    ) - 1
                                )
                                / (
                                    max_frames - 1
                                )
                            )

                            selected_indexes.append(
                                round(pos)
                            )

                        selected_segments = [
                            usable_segments[i]
                            for i in selected_indexes
                        ]

                    else:

                        selected_segments = (
                            usable_segments
                        )

                    for seg in selected_segments:

                        start = float(
                            seg.get(
                                "start",
                                0
                            )
                        )

                        end = float(
                            seg.get(
                                "end",
                                start
                            )
                        )

                        text = str(
                            seg.get(
                                "text",
                                ""
                            )
                        ).strip()

                        if not text:
                            continue

                        timestamp = (
                            start + end
                        ) / 2

                        frame_bytes = (
                            extract_frame_bytes(
                                cap,
                                timestamp
                            )
                        )

                        if frame_bytes is None:
                            continue

                        scene_items.append(
                            {
                                "start": start,
                                "end": end,
                                "timestamp": timestamp,
                                "text": text,
                                "image": frame_bytes
                            }
                        )

                    cap.release()

                    if not scene_items:

                        st.error(
                            "❌ No video frames could be extracted."
                        )

                    else:

                        prompt = """
You are analyzing an actual movie/video.

For every supplied timestamp:
LOOK AT THE ACTUAL VIDEO FRAME FIRST.

Then use the transcript only to understand the spoken words.

Return ONLY a concise scene analysis.

For each scene use exactly this format:

Scene 1 — 00:00.0 → 00:00.0
Visual: ...
Dialogue: ...

Scene 2 — 00:00.0 → 00:00.0
Visual: ...
Dialogue: ...

Rules:

- Describe ONLY what is actually visible in the supplied frame.
- Do NOT invent events.
- Do NOT invent characters.
- Do NOT invent locations.
- Do NOT invent actions.
- Do NOT add movie information from outside.
- Keep the exact chronological order.
- Keep each scene attached to its supplied timestamp.
- Do not move dialogue between timestamps.
- Do not write a long explanation.
- Do not repeat the same information.
- Do not add "Summary", "Characters", "Emotion", "Analysis"
  or other extra sections.
- If the frame does not clearly show something, do not guess it.
- If the transcript says something that is not visually confirmed,
  keep it only in Dialogue and do not describe it as a visual event.
- Keep the output short and natural.
- The purpose is to make the later recap match the actual video.
"""

                        contents = []

                        contents.append(
                            types.Part.from_text(
                                text=prompt
                            )
                        )

                        for index, item in enumerate(
                            scene_items,
                            start=1
                        ):

                            label = (
                                f"\n\n"
                                f"SCENE {index}\n"
                                f"Timestamp: "
                                f"{item['start']:.2f}s → "
                                f"{item['end']:.2f}s\n"
                                f"Transcript: "
                                f"{item['text']}\n"
                            )

                            contents.append(
                                types.Part.from_text(
                                    text=label
                                )
                            )

                            contents.append(
                                types.Part.from_bytes(
                                    data=item["image"],
                                    mime_type="image/jpeg"
                                )
                            )

                        response = (
                            client.models.generate_content(
                                model="gemini-3.6-flash",
                                contents=contents
                            )
                        )

                        scene_analysis = (
                            response.text
                            if response
                            else ""
                        )

                        if scene_analysis:

                            st.session_state[
                                "ai_scene_analysis"
                            ] = scene_analysis

                            st.session_state[
                                "scene_analysis_frames"
                            ] = len(
                                scene_items
                            )

                            st.success(
                                "✅ Scene Analysis completed."
                            )

                        else:

                            st.error(
                                "❌ Gemini returned no scene analysis."
                            )

            except Exception as e:

                st.error(
                    f"❌ Scene Analysis failed: {e}"
                )


if "ai_scene_analysis" in st.session_state:

    st.text_area(
        "🎬 Scene Analysis Result",
        st.session_state[
            "ai_scene_analysis"
        ],
        height=450
    )


st.divider()


# =========================================================
# MOVIE RECAP SCRIPT
# =========================================================

st.subheader(
    "📝 Movie Recap Script"
)


if "ai_scene_analysis" in st.session_state:

    if st.button(
        "🎬 Generate Recap Script"
    ) or run_all:

        try:

            api_key = st.secrets[
                "GEMINI_API_KEY"
            ]

            client = genai.Client(
                api_key=api_key
            )

            scene_analysis = (
                st.session_state[
                    "ai_scene_analysis"
                ]
            )

            prompt = f"""
You are a professional movie recap script writer.

Write the recap using ONLY the Scene Analysis below.

IMPORTANT:

The Scene Analysis was created by checking actual video frames.

Rules:

- Follow the exact scene order.
- Do not reorder scenes.
- Do not invent events.
- Do not invent characters.
- Do not invent locations.
- Do not invent actions.
- Do not add information from outside the video.
- Do not add events that are not in Scene Analysis.
- Do not repeat the same scene.
- Do not write extra explanation.
- Do not add headings.
- Do not add notes.
- Write only the recap narration.
- Keep the narration natural for voiceover.
- Make every sentence traceable to the actual scenes.
- If a detail is not clearly shown, leave it out.

Scene Analysis:

{scene_analysis}
"""

            response = (
                client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
            )

            recap_text = (
                response.text
                if response
                else ""
            )

            if not recap_text.strip():

                raise RuntimeError(
                    "Gemini returned an empty recap script."
                )

            st.session_state[
                "recap_script"
            ] = recap_text

            st.success(
                "✅ Movie Recap Script generated!"
            )

        except Exception as e:

            st.error(
                f"❌ Recap generation failed: {e}"
            )


if "recap_script" in st.session_state:

    st.text_area(
        "🎬 Recap Script",
        st.session_state[
            "recap_script"
        ],
        height=600
    )


st.divider()


# =========================================================
# MYANMAR RECAP SCRIPT
# =========================================================

st.subheader(
    "🇲🇲 Myanmar Recap Script"
)


if "recap_script" in st.session_state:

    if st.button(
        "🇲🇲 Translate to Myanmar"
    ) or run_all:

        try:

            api_key = st.secrets[
                "GEMINI_API_KEY"
            ]

            client = genai.Client(
                api_key=api_key
            )

            recap_script = (
                st.session_state[
                    "recap_script"
                ]
            )

            prompt = f"""
Translate the following movie recap narration
into natural spoken Myanmar Burmese.

Rules:

- Preserve the exact meaning.
- Do not add story events.
- Do not remove story events.
- Keep chronological order.
- Do not add explanation.
- Do not add English.
- Write only the Myanmar narration.
- Make it natural for voiceover.
- Use "ဒယ်" instead of "တယ်" at sentence endings
  where it sounds natural.

English Recap:

{recap_script}
"""

            response = (
                client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
            )

            myanmar_text = (
                response.text
                if response
                else ""
            )

            if not myanmar_text.strip():

                raise RuntimeError(
                    "Gemini returned an empty Myanmar recap."
                )

            st.session_state[
                "myanmar_recap"
            ] = myanmar_text

            st.success(
                "✅ Myanmar Recap Script generated!"
            )

        except Exception as e:

            st.error(
                f"❌ Myanmar translation failed: {e}"
            )


if "myanmar_recap" in st.session_state:

    st.text_area(
        "🇲🇲 Myanmar Recap",
        st.session_state[
            "myanmar_recap"
        ],
        height=300
    )


st.divider()


# =========================================================
# MYANMAR VOICEOVER
# =========================================================

st.subheader(
    "🎙️ Myanmar Voiceover"
)


if "myanmar_recap" in st.session_state:

    if voice_gender.startswith("👩"):

        selected_voice = (
            "my-MM-NilarNeural"
        )

    else:

        selected_voice = (
            "my-MM-ThihaNeural"
        )

    if float(voice_speed) == 1.0:

        tts_rate = "+0%"

    elif float(voice_speed) == 1.1:

        tts_rate = "+10%"

    else:

        tts_rate = "+20%"

    st.caption(
        "CPU optimized • "
        "Natural TTS crossfade • "
        "Actual TTS duration used for subtitle timing"
    )

    if st.button(
        "🎙️ Generate Myanmar Voiceover"
    ) or run_all:

        try:

            import edge_tts

            text = (
                st.session_state[
                    "myanmar_recap"
                ]
            ).strip()

            if not text:

                st.error(
                    "❌ Myanmar recap is empty."
                )

                st.stop()

            chunks = split_myanmar_text(
                text,
                max_chars=100
            )

            if not chunks:

                st.error(
                    "❌ Could not split Myanmar narration."
                )

                st.stop()

            st.info(
                f"📝 Narration divided into "
                f"{len(chunks)} voice segments."
            )

            work_dir = tempfile.mkdtemp(
                prefix="movie_recap_voice_"
            )

            raw_files = []

            raw_durations = []

            progress = st.progress(
                0
            )

            async def create_all_tts():

                results = []

                for index, chunk in enumerate(
                    chunks
                ):

                    raw_file = os.path.join(
                        work_dir,
                        f"raw_{index:04d}.mp3"
                    )

                    communicate = edge_tts.Communicate(
                        text=chunk,
                        voice=selected_voice,
                        rate=tts_rate,
                        volume="+0%",
                        pitch="+0Hz"
                    )

                    await communicate.save(
                        raw_file
                    )

                    results.append(
                        raw_file
                    )

                return results

            raw_files = asyncio.run(
                create_all_tts()
            )

            for index, raw_file in enumerate(
                raw_files
            ):

                duration = get_audio_duration(
                    raw_file
                )

                raw_durations.append(
                    duration
                )

                progress.progress(
                    (index + 1)
                    / len(raw_files)
                )

            TTS_CROSSFADE = 0.06

            normalized_files = []

            for index, raw_file in enumerate(
                raw_files
            ):

                normalized_file = os.path.join(
                    work_dir,
                    f"normalized_{index:04d}.wav"
                )

                normalize_result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        raw_file,
                        "-af",
                        (
                            "aformat="
                            "sample_fmts=fltp:"
                            "sample_rates=48000:"
                            "channel_layouts=stereo"
                        ),
                        normalized_file
                    ],
                    capture_output=True,
                    text=True
                )

                if normalize_result.returncode != 0:

                    raise RuntimeError(
                        normalize_result.stderr[-3000:]
                    )

                normalized_files.append(
                    normalized_file
                )

            combined_audio = os.path.join(
                work_dir,
                "combined.wav"
            )

            if len(normalized_files) == 1:

                shutil.copyfile(
                    normalized_files[0],
                    combined_audio
                )

            else:

                ffmpeg_inputs = []

                for file in normalized_files:

                    ffmpeg_inputs.extend(
                        [
                            "-i",
                            file
                        ]
                    )

                filter_parts = []

                previous_label = (
                    "[0:a]"
                )

                for i in range(
                    1,
                    len(normalized_files)
                ):

                    output_label = (
                        f"[a{i}]"
                    )

                    filter_parts.append(
                        (
                            f"{previous_label}"
                            f"[{i}:a]"
                            f"acrossfade="
                            f"d={TTS_CROSSFADE}:"
                            f"c1=tri:"
                            f"c2=tri"
                            f"{output_label}"
                        )
                    )

                    previous_label = (
                        output_label
                    )

                filter_complex = ";".join(
                    filter_parts
                )

                combine_command = [
                    "ffmpeg",
                    "-y"
                ]

                combine_command.extend(
                    ffmpeg_inputs
                )

                combine_command.extend(
                    [
                        "-filter_complex",
                        filter_complex,
                        "-map",
                        previous_label,
                        "-c:a",
                        "pcm_s16le",
                        combined_audio
                    ]
                )

                combine_result = subprocess.run(
                    combine_command,
                    capture_output=True,
                    text=True
                )

                if combine_result.returncode != 0:

                    raise RuntimeError(
                        combine_result.stderr[-3000:]
                    )

            final_voice_file = os.path.join(
                work_dir,
                "myanmar_voiceover.mp3"
            )

            convert_result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    combined_audio,
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    final_voice_file
                ],
                capture_output=True,
                text=True
            )

            if convert_result.returncode != 0:

                raise RuntimeError(
                    convert_result.stderr[-3000:]
                )

            voice_duration = get_audio_duration(
                final_voice_file
            )

            subtitle_data = []

            current_time = 0.0

            for index, chunk in enumerate(
                chunks
            ):

                raw_duration = (
                    raw_durations[index]
                )

                start_time = (
                    current_time
                )

                end_time = (
                    start_time
                    + raw_duration
                )

                if index < len(chunks) - 1:

                    next_time = (
                        end_time
                        - TTS_CROSSFADE
                    )

                else:

                    next_time = end_time

                next_time = min(
                    next_time,
                    voice_duration
                )

                if next_time <= start_time:

                    next_time = min(
                        voice_duration,
                        start_time + 0.05
                    )

                if start_time < voice_duration:

                    subtitle_data.append(
                        {
                            "start": start_time,
                            "end": next_time,
                            "text": chunk
                        }
                    )

                current_time = (
                    end_time
                    - TTS_CROSSFADE
                )

                current_time = max(
                    current_time,
                    0
                )

            st.session_state[
                "voiceover_file"
            ] = final_voice_file

            st.session_state[
                "voice_duration"
            ] = voice_duration

            st.session_state[
                "subtitle_data"
            ] = subtitle_data

            st.session_state[
                "subtitle_timing_source"
            ] = "tts_segments_crossfade"

            st.session_state[
                "voice_chunks"
            ] = chunks

            st.session_state[
                "voice_speed_value"
            ] = float(
                voice_speed
            )

            st.session_state[
                "selected_voice"
            ] = selected_voice

            st.session_state[
                "tts_rate"
            ] = tts_rate

            st.success(
                f"✅ Myanmar {voice_gender} voiceover generated!"
            )

            st.success(
                f"✅ {len(subtitle_data)} subtitles "
                "synced with the voice."
            )

            st.info(
                f"🎙️ Voiceover Duration: "
                f"{voice_duration:.2f} seconds"
            )

            st.info(
                f"🎚️ Voice Speed: "
                f"{voice_speed:.1f}×"
            )

            st.info(
                f"🗣️ Voice: "
                f"{selected_voice}"
            )

            st.info(
                "🔊 Natural TTS crossfade: "
                f"{TTS_CROSSFADE:.2f} sec"
            )

        except Exception as e:

            st.error(
                f"❌ Voiceover generation failed: {e}"
            )


# =========================================================
# VOICEOVER PLAYER
# =========================================================

if "voiceover_file" in st.session_state:

    st.audio(
        st.session_state[
            "voiceover_file"
        ],
        format="audio/mp3"
    )


st.divider()


# =========================================================
# MYANMAR SUBTITLE
# =========================================================

st.subheader(
    "🇲🇲 Myanmar Subtitle"
)


if "subtitle_data" in st.session_state:

    st.success(
        "✅ Subtitle timing is synchronized "
        "with the generated voiceover."
    )

    preview_text = "\n".join(
        [
            f'{item["start"]:.2f}s → '
            f'{item["end"]:.2f}s | '
            f'{item["text"]}'
            for item in st.session_state[
                "subtitle_data"
            ]
        ]
    )

    st.text_area(
        "💬 Myanmar Subtitle Preview",
        preview_text,
        height=400
    )

    st.caption(
        "ℹ️ Subtitle timing comes directly "
        "from the TTS segments."
    )


st.divider()


# =========================================================
# SUBTITLE EXPORT
# =========================================================

st.subheader(
    "📥 Subtitle Export"
)


if "subtitle_data" in st.session_state:

    srt_lines = []

    for i, item in enumerate(
        st.session_state[
            "subtitle_data"
        ],
        start=1
    ):

        start = float(
            item["start"]
        )

        end = float(
            item["end"]
        )

        text = str(
            item["text"]
        ).strip()

        start_h = int(
            start // 3600
        )

        start_m = int(
            (start % 3600) // 60
        )

        start_s = int(
            start % 60
        )

        start_ms = int(
            (start % 1) * 1000
        )

        end_h = int(
            end // 3600
        )

        end_m = int(
            (end % 3600) // 60
        )

        end_s = int(
            end % 60
        )

        end_ms = int(
            (end % 1) * 1000
        )

        start_time = (
            f"{start_h:02d}:"
            f"{start_m:02d}:"
            f"{start_s:02d},"
            f"{start_ms:03d}"
        )

        end_time = (
            f"{end_h:02d}:"
            f"{end_m:02d}:"
            f"{end_s:02d},"
            f"{end_ms:03d}"
        )

        srt_lines.append(
            f"{i}\n"
            f"{start_time} --> "
            f"{end_time}\n"
            f"{text}\n"
        )

    srt_content = "\n".join(
        srt_lines
    )

    st.download_button(
        "📥 Download Myanmar Subtitle (.srt)",
        data=srt_content.encode(
            "utf-8"
        ),
        file_name="myanmar_subtitles.srt",
        mime="text/plain"
    )


st.divider()


# =========================================================
# VOICEOVER TIMING
# =========================================================

st.subheader(
    "⏱️ Voiceover Timing"
)


if "voiceover_file" in st.session_state:

    try:

        voice_duration = get_audio_duration(
            st.session_state[
                "voiceover_file"
            ]
        )

        st.session_state[
            "voice_duration"
        ] = voice_duration

        st.write(
            f"🎙️ Voiceover Duration: "
            f"{voice_duration:.2f} seconds"
        )

    except Exception as e:

        st.warning(
            f"⚠️ Could not read voice duration: {e}"
        )


st.divider()


# =========================================================
# SCENE TIMING
# =========================================================

st.subheader(
    "⏱️ Scene Timing"
)


if "subtitle_data" in st.session_state:

    subtitle_duration = st.session_state.get(
        "voice_duration",
        0
    )

    fixed_subtitles = []

    for item in st.session_state[
        "subtitle_data"
    ]:

        start = max(
            0,
            float(
                item["start"]
            )
        )

        end = float(
            item["end"]
        )

        text = str(
            item["text"]
        ).strip()

        if not text:
            continue

        if subtitle_duration > 0:

            if start >= subtitle_duration:

                continue

            end = min(
                end,
                subtitle_duration
            )

        if end <= start:

            continue

        fixed_subtitles.append(
            {
                "start": start,
                "end": end,
                "text": text
            }
        )

    st.session_state[
        "subtitle_data"
    ] = fixed_subtitles

    st.session_state[
        "subtitle_timing_source"
    ] = "tts_segments_crossfade"

    st.success(
        f"✅ Voiceover synced: "
        f"{len(fixed_subtitles)} subtitles"
    )


st.divider()


# =========================================================
# FREEZE FRAME + ZOOM
# =========================================================

st.subheader(
    "🧊 Freeze Frame + Zoom"
)


st.write(
    f"Status: "
    f"{'ON' if freeze_enabled else 'OFF'}"
)


if freeze_enabled:

    st.write(
        f"⏱️ Freeze Every: "
        f"{freeze_interval:.1f} seconds"
    )

    st.write(
        f"🧊 Freeze Duration: "
        f"{freeze_duration:.1f} seconds"
    )

    st.caption(
        "Every selected interval → "
        "Freeze + Zoom In → Zoom Out"
    )


st.divider()


# =========================================================
# FINAL VIDEO EXPORT
# =========================================================

st.subheader(
    "🎬 Final Video Export"
)


if (
    "voiceover_file" in st.session_state
    and
    "subtitle_data" in st.session_state
    and
    "uploaded_file" in st.session_state
):

    if st.button(
        "🎬 Create Final Recap Video"
    ) or run_all:

        try:

            # =================================================
            # SAVE ORIGINAL VIDEO
            # =================================================

            original_video = os.path.join(
                tempfile.gettempdir(),
                "movie_recap_original.mp4"
            )

            with open(
                original_video,
                "wb"
            ) as f:

                f.write(
                    st.session_state[
                        "uploaded_file"
                    ]
                )

            voice_file = (
                st.session_state[
                    "voiceover_file"
                ]
            )

            # =================================================
            # VIDEO DURATION
            # =================================================

            video_duration = get_audio_duration(
                original_video
            )

            # =================================================
            # VOICEOVER DURATION
            # =================================================

            voice_duration = get_audio_duration(
                voice_file
            )

            st.session_state[
                "voice_duration"
            ] = voice_duration

            st.info(
                f"🎙️ Voiceover: "
                f"{voice_duration:.2f} sec"
            )

            # =================================================
            # FREEZE SETTINGS
            # =================================================

            if freeze_enabled:

                interval = float(
                    freeze_interval
                )

                freeze_time = float(
                    freeze_duration
                )

            else:

                interval = (
                    video_duration + 1
                )

                freeze_time = 0

            # =================================================
            # OUTPUT DIMENSIONS / FONT SETTINGS
            # =================================================

            original_width = int(
                st.session_state.get(
                    "video_width",
                    576
                )
            )

            original_height = int(
                st.session_state.get(
                    "video_height",
                    1024
                )
            )

            if aspect_ratio == "Original":

                output_width = max(
                    2,
                    original_width
                    - (
                        original_width % 2
                    )
                )

                output_height = max(
                    2,
                    original_height
                    - (
                        original_height % 2
                    )
                )

            elif aspect_ratio == "9:16":

                output_width = 576
                output_height = 1024

            elif aspect_ratio == "16:9":

                output_width = 1024
                output_height = 576

            elif aspect_ratio == "1:1":

                output_width = 768
                output_height = 768

            else:

                output_width = 768
                output_height = 1024

            selected_font_name = (
                subtitle_font
            )

            font_dir = None

            if (
                subtitle_font == "Custom Font"
                and
                custom_font_file is not None
            ):

                font_dir = os.path.join(
                    tempfile.gettempdir(),
                    "movie_recap_fonts"
                )

                os.makedirs(
                    font_dir,
                    exist_ok=True
                )

                custom_font_path = os.path.join(
                    font_dir,
                    custom_font_file.name
                )

                with open(
                    custom_font_path,
                    "wb"
                ) as font_handle:

                    font_handle.write(
                        custom_font_file.getvalue()
                    )

                selected_font_name = (
                    "CustomFont"
                )

            subtitle_wrap_chars = (
                24
                if output_height >= output_width
                else 42
            )

            subtitle_margin_v = max(
                40,
                int(
                    output_height * 0.055
                )
            )

            # =================================================
            # ASS SUBTITLES
            # =================================================

            ass_content = (
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                f"PlayResX: {output_width}\n"
                f"PlayResY: {output_height}\n"
                "ScaledBorderAndShadow: yes\n\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, "
                "PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, "
                "Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, "
                "MarginV, Encoding\n"
                f"Style: Myanmar,"
                f"{selected_font_name},"
                f"{subtitle_size},"
                f"&H0000FFFF,"
                f"&H0000FFFF,"
                f"&H00000000,"
                f"&H99000000,"
                f"0,0,0,0,"
                f"100,100,0,0,"
                f"1,2,1,2,"
                f"40,40,"
                f"{subtitle_margin_v},1\n\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, "
                "Name, MarginL, MarginR, MarginV, "
                "Effect, Text\n"
            )

            for item in st.session_state[
                "subtitle_data"
            ]:

                new_start = max(
                    0,
                    float(
                        item["start"]
                    )
                )

                new_end = max(
                    new_start + 0.1,
                    float(
                        item["end"]
                    )
                )

                text = wrap_myanmar(
                    item["text"],
                    subtitle_wrap_chars
                )

                text = text.replace(
                    "{",
                    "\\{"
                )

                text = text.replace(
                    "}",
                    "\\}"
                )

                ass_content += (
                    f"Dialogue: 0,"
                    f"{ass_time(new_start)},"
                    f"{ass_time(new_end)},"
                    f"Myanmar,,0,0,0,,"
                    f"{text}\n"
                )

            ass_file = os.path.join(
                tempfile.gettempdir(),
                "myanmar_subtitles.ass"
            )

            with open(
                ass_file,
                "w",
                encoding="utf-8-sig"
            ) as f:

                f.write(
                    ass_content
                )

            # =================================================
            # BLUR SETTINGS
            # =================================================

            if blur_enabled:

                blur_x_value = float(
                    st.session_state.get(
                        "main_blur_x_value",
                        10.0
                    )
                )

                blur_y_value = float(
                    st.session_state.get(
                        "main_blur_y_value",
                        78.0
                    )
                )

                blur_width_value = float(
                    st.session_state.get(
                        "main_blur_width_value",
                        80.0
                    )
                )

                blur_height_value = float(
                    st.session_state.get(
                        "main_blur_height_value",
                        18.0
                    )
                )

                # -------------------------------------------------
                # Final output pixel coordinates
                # -------------------------------------------------

                blur_x_px = int(
                    output_width
                    * blur_x_value
                    / 100.0
                )

                blur_y_px = int(
                    output_height
                    * blur_y_value
                    / 100.0
                )

                blur_width_px = int(
                    output_width
                    * blur_width_value
                    / 100.0
                )

                blur_height_px = int(
                    output_height
                    * blur_height_value
                    / 100.0
                )

                # Force even dimensions

                blur_width_px = max(
                    2,
                    blur_width_px
                    - blur_width_px % 2
                )

                blur_height_px = max(
                    2,
                    blur_height_px
                    - blur_height_px % 2
                )

                blur_width_px = min(
                    blur_width_px,
                    output_width
                )

                blur_height_px = min(
                    blur_height_px,
                    output_height
                )

                blur_x_px = max(
                    0,
                    min(
                        blur_x_px,
                        output_width
                        - blur_width_px
                    )
                )

                blur_y_px = max(
                    0,
                    min(
                        blur_y_px,
                        output_height
                        - blur_height_px
                    )
                )

                blur_strength_value = max(
                    1,
                    int(
                        blur_strength
                    )
                )

                st.info(
                    f"🔲 Final Blur: "
                    f"X {blur_x_value:.1f}% | "
                    f"Y {blur_y_value:.1f}% | "
                    f"W {blur_width_value:.1f}% | "
                    f"H {blur_height_value:.1f}%"
                )

            # =================================================
            # VIDEO FILTER
            # =================================================

            filter_parts = []

            labels = []

            # -------------------------------------------------
            # Helper: blur a scaled video stream
            #
            # Blur is applied BEFORE zoompan.
            # Therefore the blur area belongs to the freeze
            # frame and zooms together with the frame.
            # -------------------------------------------------

            def make_blur_filter(
                input_label,
                output_label
            ):

                if not blur_enabled:

                    filter_parts.append(
                        f"{input_label}"
                        f"format=yuv420p"
                        f"{output_label}"
                    )

                    return

                unique_name = (
                    output_label
                    .strip("[]")
                    .replace(
                        "-",
                        "_"
                    )
                )

                blurbase_name = (
                    f"[blurbase_{unique_name}]"
                )

                cleanbase_name = (
                    f"[cleanbase_{unique_name}]"
                )

                blurred_name = (
                    f"[blurred_{unique_name}]"
                )

                filter_parts.append(
                    f"{input_label}"
                    f"split=2"
                    f"{blurbase_name}"
                    f"{cleanbase_name}"
                )

                filter_parts.append(
                    blurbase_name
                    + f"crop="
                    f"{blur_width_px}:"
                    f"{blur_height_px}:"
                    f"{blur_x_px}:"
                    f"{blur_y_px},"
                    f"boxblur="
                    f"luma_radius={blur_strength_value}:"
                    f"luma_power=2:"
                    f"chroma_radius={blur_strength_value}:"
                    f"chroma_power=2"
                    + blurred_name
                )

                filter_parts.append(
                    cleanbase_name
                    + blurred_name
                    + f"overlay="
                    f"{blur_x_px}:"
                    f"{blur_y_px}"
                    + output_label
                )

            # -------------------------------------------------
            # Build Normal + Freeze segments
            # -------------------------------------------------

            if freeze_enabled:

                segment_count = int(
                    math.ceil(
                        video_duration
                        / interval
                    )
                )

                for i in range(
                    segment_count
                ):

                    start = (
                        i * interval
                    )

                    end = min(
                        (i + 1) * interval,
                        video_duration
                    )

                    normal_label = (
                        f"[normal{i}]"
                    )

                    normal_scaled_label = (
                        f"[normal_scaled{i}]"
                    )

                    # -------------------------------------------------
                    # Normal segment -> scale first
                    # -------------------------------------------------

                    filter_parts.append(
                        f"[0:v]"
                        f"trim="
                        f"start={start}:"
                        f"end={end},"
                        f"setpts=PTS-STARTPTS,"
                        f"scale="
                        f"{output_width}:"
                        f"{output_height},"
                        f"setsar=1"
                        f"{normal_scaled_label}"
                    )

                    # -------------------------------------------------
                    # Apply blur
                    # -------------------------------------------------

                    make_blur_filter(
                        normal_scaled_label,
                        normal_label
                    )

                    labels.append(
                        normal_label
                    )

                    # =============================================
                    # FREEZE + ZOOM
                    # =============================================

                    if end < video_duration:

                        freeze_source_label = (
                            f"[freeze_source{i}]"
                        )

                        freeze_label = (
                            f"[freeze{i}]"
                        )

                        frame_time = max(
                            start,
                            end - 0.10
                        )

                        freeze_frames = max(
                            2,
                            int(
                                round(
                                    freeze_time * 30
                                )
                            )
                        )

                        half_frames = max(
                            1,
                            freeze_frames // 2
                        )

                        if freeze_frames <= 2:

                            zoom_expression = "1"

                        else:

                            zoom_in_end = (
                                half_frames - 1
                            )

                            zoom_out_frames = max(
                                1,
                                freeze_frames
                                - half_frames
                                - 1
                            )

                            zoom_expression = (
                                f"if("
                                f"lte(on,{zoom_in_end}),"
                                f"1+0.15*on/"
                                f"{max(1, zoom_in_end)},"
                                f"1.15-0.15*"
                                f"(on-{half_frames})/"
                                f"{zoom_out_frames}"
                                f")"
                            )

                        # -----------------------------------------
                        # Extract ONE actual frame
                        # Scale to final size
                        # -----------------------------------------

                        filter_parts.append(
                            f"[0:v]"
                            f"trim="
                            f"start={frame_time}:"
                            f"end={frame_time + 0.0334},"
                            f"setpts=PTS-STARTPTS,"
                            f"select='eq(n,0)',"
                            f"scale="
                            f"{output_width}:"
                            f"{output_height},"
                            f"setsar=1"
                            f"{freeze_source_label}"
                        )

                        # -----------------------------------------
                        # BLUR BEFORE ZOOM
                        #
                        # This is important:
                        # blur is part of the freeze frame.
                        # Therefore ZoomPan zooms the blurred
                        # freeze frame together with the box.
                        # -----------------------------------------

                        freeze_blur_label = (
                            f"[freeze_blur{i}]"
                        )

                        if blur_enabled:

                            filter_parts.append(
                                freeze_source_label
                                + "split=2"
                                f"[freeze_blurbase{i}]"
                                f"[freeze_cleanbase{i}]"
                            )

                            filter_parts.append(
                                f"[freeze_blurbase{i}]"
                                f"crop="
                                f"{blur_width_px}:"
                                f"{blur_height_px}:"
                                f"{blur_x_px}:"
                                f"{blur_y_px},"
                                f"boxblur="
                                f"luma_radius="
                                f"{blur_strength_value}:"
                                f"luma_power=2:"
                                f"chroma_radius="
                                f"{blur_strength_value}:"
                                f"chroma_power=2"
                                f"[freeze_blurredarea{i}]"
                            )

                            filter_parts.append(
                                f"[freeze_cleanbase{i}]"
                                f"[freeze_blurredarea{i}]"
                                f"overlay="
                                f"{blur_x_px}:"
                                f"{blur_y_px}"
                                f"{freeze_blur_label}"
                            )

                        else:

                            filter_parts.append(
                                freeze_source_label
                                + freeze_blur_label
                            )

                        # -----------------------------------------
                        # Zoom blurred freeze frame
                        # -----------------------------------------

                        filter_parts.append(
                            freeze_blur_label
                            + f"zoompan="
                            f"z='{zoom_expression}':"
                            f"d={freeze_frames}:"
                            f"x='iw/2-(iw/zoom/2)':"
                            f"y='ih/2-(ih/zoom/2)':"
                            f"s="
                            f"{output_width}x"
                            f"{output_height}:"
                            f"fps=30"
                            + freeze_label
                        )

                        labels.append(
                            freeze_label
                        )

                concat_inputs = "".join(
                    labels
                )

                concat_filter = (
                    f"{concat_inputs}"
                    f"concat="
                    f"n={len(labels)}:"
                    f"v=1:"
                    f"a=0:"
                    f"unsafe=1,"
                    f"format=yuv420p"
                    f"[basevideo]"
                )

                filter_parts.append(
                    concat_filter
                )

            else:

                if blur_enabled:

                    filter_parts.append(
                        "[0:v]"
                        f"scale="
                        f"{output_width}:"
                        f"{output_height},"
                        "setsar=1"
                        "[scaledvideo]"
                    )

                    filter_parts.append(
                        "[scaledvideo]"
                        "split=2"
                        "[blurbase_single]"
                        "[cleanbase_single]"
                    )

                    filter_parts.append(
                        "[blurbase_single]"
                        f"crop="
                        f"{blur_width_px}:"
                        f"{blur_height_px}:"
                        f"{blur_x_px}:"
                        f"{blur_y_px},"
                        f"boxblur="
                        f"luma_radius="
                        f"{blur_strength_value}:"
                        f"luma_power=2:"
                        f"chroma_radius="
                        f"{blur_strength_value}:"
                        f"chroma_power=2"
                        "[blurredarea_single]"
                    )

                    filter_parts.append(
                        "[cleanbase_single]"
                        "[blurredarea_single]"
                        f"overlay="
                        f"{blur_x_px}:"
                        f"{blur_y_px}"
                        "[basevideo]"
                    )

                else:

                    filter_parts.append(
                        "[0:v]"
                        f"scale="
                        f"{output_width}:"
                        f"{output_height},"
                        "setsar=1,"
                        "format=yuv420p"
                        "[basevideo]"
                    )

            # =================================================
            # SUBTITLE OVERLAY
            # =================================================

            ass_filter = (
                f"ass={ass_file}"
            )

            if font_dir:

                ass_filter += (
                    f":fontsdir={font_dir}"
                )

            filter_parts.append(
                "[basevideo]"
                + ass_filter
                + "[vout]"
            )

            # =================================================
            # DURATION CALCULATION
            # =================================================

            if freeze_enabled:

                actual_freeze_count = max(
                    0,
                    segment_count - 1
                )

            else:

                actual_freeze_count = 0

            base_video_duration = (
                video_duration
                +
                (
                    actual_freeze_count
                    * freeze_time
                )
            )

            extra_duration = max(
                0.0,
                voice_duration
                - base_video_duration
            )

            st.write(
                f"🎬 Base Video Duration: "
                f"{base_video_duration:.2f} sec"
            )

            st.write(
                f"➕ Extra Hold Duration: "
                f"{extra_duration:.2f} sec"
            )

            # =================================================
            # EXTEND FINAL FRAME
            # =================================================

            if extra_duration > 0:

                filter_parts.append(
                    "[vout]"
                    f"tpad="
                    f"stop_mode=clone:"
                    f"stop_duration="
                    f"{extra_duration:.3f}:"
                    f"start_mode=clone,"
                    f"setpts=PTS-STARTPTS"
                    "[vout2]"
                )

                final_video_label = (
                    "[vout2]"
                )

            else:

                final_video_label = (
                    "[vout]"
                )

            # =================================================
            # FILTER COMPLEX
            # =================================================

            filter_complex = ";".join(
                filter_parts
            )

            # =================================================
            # OUTPUT
            # =================================================

            output_video = os.path.join(
                tempfile.gettempdir(),
                "final_movie_recap.mp4"
            )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                original_video,
                "-i",
                voice_file,
                "-filter_complex",
                filter_complex,
                "-map",
                final_video_label,
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "27",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-t",
                f"{voice_duration:.3f}",
                output_video
            ]

            # =================================================
            # EXPORT
            # =================================================

            st.info(
                "⏳ Creating final video..."
            )

            if blur_enabled:

                st.info(
                    "🔲 Original subtitle blur is enabled."
                )

                st.info(
                    "🔍 Blur is attached before Freeze + Zoom."
                )

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:

                st.error(
                    "❌ FFmpeg Export Failed"
                )

                st.code(
                    result.stderr[-5000:],
                    language="text"
                )

            else:

                final_duration = get_audio_duration(
                    output_video
                )

                st.success(
                    "✅ Final Recap Video "
                    "Created Successfully!"
                )

                st.info(
                    f"🎬 Final Video Duration: "
                    f"{final_duration:.2f} sec"
                )

                st.info(
                    f"🎙️ Voiceover Duration: "
                    f"{voice_duration:.2f} sec"
                )

                duration_difference = (
                    final_duration
                    - voice_duration
                )

                st.info(
                    f"⏱️ Duration Difference: "
                    f"{duration_difference:+.2f} sec"
                )

                with open(
                    output_video,
                    "rb"
                ) as f:

                    st.download_button(
                        "⬇️ Download Final Video",
                        f.read(),
                        file_name=(
                            "final_movie_recap.mp4"
                        ),
                        mime="video/mp4"
                    )

        except Exception as e:

            st.error(
                f"❌ Final Video Export Error: {e}"
            )
