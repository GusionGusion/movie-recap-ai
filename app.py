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
    max_chars=14
):
    """
    Myanmar subtitle wrapping.

    Rules:
    - Short text = 1 line
    - Long text = maximum 2 lines
    - Never intentionally create 3 or 4 lines
    - Designed for large subtitle font sizes
    """

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if not text:
        return ""

    # -----------------------------------------------------
    # Short subtitle → one line
    # -----------------------------------------------------

    if len(text) <= max_chars:

        return text


    # -----------------------------------------------------
    # Try word-based balanced 2-line split
    # -----------------------------------------------------

    words = text.split()

    if len(words) <= 1:

        midpoint = max(
            1,
            len(text) // 2
        )

        return (
            text[:midpoint]
            + "\\N"
            + text[midpoint:]
        )


    total_length = len(text)

    target = total_length / 2

    best_split = None

    best_difference = None


    for split_index in range(
        1,
        len(words)
    ):

        line1 = " ".join(
            words[:split_index]
        ).strip()

        line2 = " ".join(
            words[split_index:]
        ).strip()

        if not line1 or not line2:

            continue

        difference = abs(
            len(line1) - len(line2)
        )

        if (
            best_difference is None
            or difference < best_difference
        ):

            best_difference = difference

            best_split = split_index


    if best_split is None:

        midpoint = max(
            1,
            len(text) // 2
        )

        return (
            text[:midpoint]
            + "\\N"
            + text[midpoint:]
        )


    line1 = " ".join(
        words[:best_split]
    ).strip()

    line2 = " ".join(
        words[best_split:]
    ).strip()


    if not line2:

        return line1


    return (
        line1
        + "\\N"
        + line2
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
# CACHED WHISPER MODEL
# =========================================================

@st.cache_resource
def load_whisper_model(
    model_name="base"
):

    import whisper

    return whisper.load_model(
        model_name
    )


# =========================================================
# VIDEO SPLIT + MERGE HELPERS
# =========================================================

SPLIT_DURATION = 60.0


def split_video_into_parts(
    input_video,
    output_dir,
    split_duration=60.0
):
    """
    Split a video into 60-second parts.

    This is FFmpeg-only processing.
    Gemini API is NOT called here.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    duration = get_audio_duration(
        input_video
    )

    if duration <= split_duration:

        return [
            input_video
        ]

    part_count = int(
        math.ceil(
            duration / split_duration
        )
    )

    part_files = []

    for index in range(
        part_count
    ):

        start_time = (
            index * split_duration
        )

        part_file = os.path.join(
            output_dir,
            f"part_{index + 1:03d}.mp4"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            input_video,
            "-t",
            f"{split_duration:.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            part_file
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            # Fallback to accurate re-encoding
            fallback_command = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_time:.3f}",
                "-i",
                input_video,
                "-t",
                f"{split_duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
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
                part_file
            ]

            fallback_result = subprocess.run(
                fallback_command,
                capture_output=True,
                text=True
            )

            if fallback_result.returncode != 0:

                raise RuntimeError(
                    fallback_result.stderr[-3000:]
                )

        if os.path.isfile(
            part_file
        ):

            part_files.append(
                part_file
            )

    return part_files


def merge_video_parts(
    part_files,
    output_file
):
    """
    Merge already rendered video parts.

    Gemini API is NOT called.
    """

    if not part_files:

        raise RuntimeError(
            "No video parts available for merge."
        )


    if len(part_files) == 1:

        shutil.copyfile(
            part_files[0],
            output_file
        )

        return output_file


    concat_file = os.path.join(
        os.path.dirname(output_file),
        "movie_recap_concat.txt"
    )


    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:

        for part in part_files:

            safe_path = (
                os.path.abspath(part)
                .replace(
                    "\\",
                    "/"
                )
                .replace(
                    "'",
                    "'\\''"
                )
            )

            f.write(
                f"file '{safe_path}'\n"
            )


    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c",
        "copy",
        output_file
    ]


    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )


    if result.returncode != 0:

        # Fallback if stream copy cannot concatenate
        fallback_command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
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
            output_file
        ]

        fallback_result = subprocess.run(
            fallback_command,
            capture_output=True,
            text=True
        )

        if fallback_result.returncode != 0:

            raise RuntimeError(
                fallback_result.stderr[-5000:]
            )


    return output_file


def split_and_merge_final_video(
    input_video,
    work_dir,
    split_duration=60.0
):
    """
    Final export split mode.

    1. Check duration.
    2. If <= 60 sec, return original.
    3. If > 60 sec, split into 60-sec parts.
    4. Merge all parts back into one final video.

    No Gemini API calls are made.
    """

    duration = get_audio_duration(
        input_video
    )

    if duration <= split_duration:

        return input_video, 1


    split_dir = os.path.join(
        work_dir,
        "split_parts"
    )

    os.makedirs(
        split_dir,
        exist_ok=True
    )


    part_files = split_video_into_parts(
        input_video,
        split_dir,
        split_duration
    )


    if len(part_files) <= 1:

        return input_video, 1


    merged_file = os.path.join(
        work_dir,
        "final_movie_recap_merged.mp4"
    )


    merge_video_parts(
        part_files,
        merged_file
    )


    return merged_file, len(part_files)


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

        # =====================================================
        # AUTO SPLIT INFORMATION
        # =====================================================

        if duration_seconds > SPLIT_DURATION:

            split_count = int(
                math.ceil(
                    duration_seconds
                    / SPLIT_DURATION
                )
            )

            st.info(
                f"✂️ Split Mode: ON • "
                f"Video will be handled in "
                f"{split_count} × 60-second parts "
                f"before final merge."
            )

        else:

            st.info(
                "✂️ Split Mode: Not required "
                "for videos 60 seconds or shorter."
            )

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


# =========================================================
# SUBTITLE FONT SETTINGS
# =========================================================

st.subheader(
    "🔤 Subtitle Font Settings"
)


FONT_OPTIONS = {
    "Noto Sans Myanmar": {
        "font_name": "Noto Sans Myanmar",
        "file_name": "NotoSansMyanmar-Regular.ttf"
    },

    "Noto Serif Myanmar": {
        "font_name": "Noto Serif Myanmar",
        "file_name": "NotoSerifMyanmar-Regular.ttf"
    },

    "Padauk": {
        "font_name": "Padauk",
        "file_name": "Padauk-Regular.ttf"
    }
}


subtitle_font = st.selectbox(
    "🔤 Subtitle Font",
    list(FONT_OPTIONS.keys()),
    index=0,
    help=(
        "Choose a Myanmar Unicode font for the final subtitle."
    ),
    key="main_subtitle_font"
)


selected_font_info = FONT_OPTIONS[
    subtitle_font
]


selected_font_name = selected_font_info[
    "font_name"
]


fonts_dir = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "fonts"
)


selected_font_file = os.path.join(
    fonts_dir,
    selected_font_info["file_name"]
)


if os.path.isfile(
    selected_font_file
):

    st.success(
        f"✅ Font ready: {subtitle_font}"
    )

else:

    st.warning(
        f"⚠️ {subtitle_font} font file was not found. "
        f"Please add {selected_font_info['file_name']} "
        f"inside the fonts/ folder."
    )


subtitle_font_size = st.number_input(
    "📏 Subtitle Font Size (Manual)",
    min_value=20,
    max_value=100,
    value=50,
    step=1,
    help="Default subtitle font size is 50px."
)


st.caption(
    "🟡 Subtitle Color: Yellow"
)

st.caption(
    "⬛ Subtitle Outline: Black"
)

st.caption(
    "Default: 50px • Subtitle maximum: 2 lines"
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
            f"⏳ Transcribing with Whisper "
            f"{whisper_model}..."
        )

        try:

            model = load_whisper_model(
                whisper_model
            )

            result = model.transcribe(
                st.session_state[
                    "video_path"
                ],
                language="en"
            )

            st.session_state[
                "transcript_result"
            ] = result

            st.session_state[
                "transcript"
            ] = result["text"]

            st.success(
                "✅ Transcript generated!"
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

            st.session_state[
                "recap_script"
            ] = response.text

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
- Keep approximately the same amount of information as the English recap.
- Do not unnecessarily expand the narration.
- Do not add extra sentences just to make the translation longer.

English Recap:

{recap_script}
"""

            response = (
                client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
            )

            myanmar_text = response.text

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


            # =================================================
            # TTS GENERATION
            # =================================================

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


            # =================================================
            # MEASURE ACTUAL DURATIONS
            # =================================================

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


            # =================================================
            # NATURAL CROSSFADE
            # =================================================

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


            # =================================================
            # COMBINE WITH ACROSSFADE
            # =================================================

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

                previous_label = "[0:a]"


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


            # =================================================
            # FINAL MP3
            # =================================================

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


            # =================================================
            # ACTUAL FINAL DURATION
            # =================================================

            voice_duration = get_audio_duration(
                final_voice_file
            )


            # =================================================
            # SUBTITLE TIMING
            # =================================================

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


            # =================================================
            # SESSION STATE
            # =================================================

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
                f"synced with the voice."
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
        "ℹ️ Subtitle: short = 1 line, "
        "long = maximum 2 lines."
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

    # =====================================================
    # SPLIT MODE SETTINGS
    # =====================================================

    st.caption(
        "✂️ Videos longer than 1 minute are automatically "
        "exported in 60-second parts and combined again."
    )

    split_mode = st.checkbox(
        "✂️ Enable 60-Second Split Export",
        value=True,
        key="split_export_mode"
    )

    split_duration = 60.0


    if split_mode:

        st.info(
            "✂️ Split Mode ON — "
            "Each part will be exported for 60 seconds "
            "and combined into one final video."
        )

    else:

        st.info(
            "🎬 Split Mode OFF — "
            "Video will be exported normally."
        )


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
            # ASS SUBTITLE SETTINGS
            # =================================================

            font_size = int(
                subtitle_font_size
            )


            subtitle_wrap_chars = max(
                10,
                int(
                    24
                    * 28
                    / max(
                        font_size,
                        1
                    )
                )
            )


            # =================================================
            # TEMP EXPORT DIRECTORY
            # =================================================

            export_dir = tempfile.mkdtemp(
                prefix="movie_recap_split_export_"
            )


            part_files = []


            # =================================================
            # DETERMINE SPLIT COUNT
            # =================================================

            if split_mode and video_duration > split_duration:

                part_count = int(
                    math.ceil(
                        video_duration
                        / split_duration
                    )
                )

            else:

                part_count = 1


            st.write(
                f"🎬 Export Parts: "
                f"{part_count}"
            )


            # =================================================
            # CREATE EACH PART
            # =================================================

            for part_index in range(
                part_count
            ):

                part_start = (
                    part_index
                    * split_duration
                    if part_count > 1
                    else 0
                )

                part_end = min(
                    part_start + split_duration,
                    video_duration
                )


                part_video_duration = (
                    part_end
                    - part_start
                )


                st.info(
                    f"⏳ Exporting Part "
                    f"{part_index + 1}/{part_count} "
                    f"("
                    f"{part_start:.1f}s → "
                    f"{part_end:.1f}s"
                    f")"
                )


                # =================================================
                # LOCAL SUBTITLE ASS FILE
                # =================================================

                ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 576
PlayResY: 1024
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Myanmar,{selected_font_name},{font_size},&H0000FFFF,&H0000FFFF,&H00000000,&H99000000,0,0,0,0,100,95,0,0,1,2,0,2,30,30,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


                # =================================================
                # SUBTITLES BELONGING TO THIS PART
                # =================================================

                for item in st.session_state[
                    "subtitle_data"
                ]:

                    global_start = max(
                        0,
                        float(
                            item["start"]
                        )
                    )

                    global_end = max(
                        global_start + 0.1,
                        float(
                            item["end"]
                        )
                    )


                    # ---------------------------------------------
                    # Skip subtitles outside this part
                    # ---------------------------------------------

                    if global_end <= part_start:

                        continue

                    if global_start >= part_end:

                        continue


                    # ---------------------------------------------
                    # Convert global timing → local timing
                    # ---------------------------------------------

                    local_start = max(
                        0,
                        global_start - part_start
                    )

                    local_end = min(
                        part_video_duration,
                        global_end - part_start
                    )


                    if local_end <= local_start:

                        continue


                    # ---------------------------------------------
                    # Wrap Myanmar subtitle
                    # ---------------------------------------------

                    text = wrap_myanmar(
                        item["text"],
                        subtitle_wrap_chars
                    )


                    # ---------------------------------------------
                    # ASS escaping
                    # ---------------------------------------------

                    text = text.replace(
                        "{",
                        "\\{"
                    )

                    text = text.replace(
                        "}",
                        "\\}"
                    )


                    text = (
                        "{\\q2}"
                        + text
                    )


                    ass_content += (
                        f"Dialogue: 0,"
                        f"{ass_time(local_start)},"
                        f"{ass_time(local_end)},"
                        f"Myanmar,,0,0,0,,"
                        f"{text}\n"
                    )


                # =================================================
                # SAVE PART ASS
                # =================================================

                ass_file = os.path.join(
                    export_dir,
                    f"part_{part_index:03d}.ass"
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
                # ESCAPE ASS PATH
                # =================================================

                ass_filter_file = (
                    ass_file
                    .replace("\\", "/")
                    .replace("'", "\\'")
                )


                ass_fonts_dir = (
                    fonts_dir
                    .replace("\\", "/")
                    .replace("'", "\\'")
                )


                # =================================================
                # PART VIDEO FILTER
                # =================================================

                filter_parts = []

                labels = []


                # =================================================
                # FREEZE + ZOOM FOR THIS PART
                # =================================================

                if freeze_enabled:

                    local_segment_count = int(
                        math.ceil(
                            part_video_duration
                            / interval
                        )
                    )


                    for local_index in range(
                        local_segment_count
                    ):

                        local_start = (
                            local_index
                            * interval
                        )

                        local_end = min(
                            (local_index + 1)
                            * interval,
                            part_video_duration
                        )


                        if local_end <= local_start:

                            continue


                        normal_label = (
                            f"normal_{part_index}_{local_index}"
                        )


                        absolute_start = (
                            part_start
                            + local_start
                        )

                        absolute_end = (
                            part_start
                            + local_end
                        )


                        normal_filter = (
                            f"[0:v]"
                            f"trim="
                            f"start={absolute_start}:"
                            f"end={absolute_end},"
                            f"setpts=PTS-STARTPTS,"
                            f"scale=576:1024,"
                            f"setsar=1"
                            f"[{normal_label}]"
                        )


                        filter_parts.append(
                            normal_filter
                        )


                        labels.append(
                            f"[{normal_label}]"
                        )


                        # =========================================
                        # FREEZE FRAME + ZOOM
                        # =========================================

                        if (
                            local_end
                            <
                            part_video_duration
                        ):

                            freeze_label = (
                                f"freeze_{part_index}_{local_index}"
                            )


                            frame_time = max(
                                absolute_start,
                                absolute_end - 0.10
                            )


                            freeze_filter = (
                                f"[0:v]"
                                f"trim="
                                f"start={frame_time}:"
                                f"end={frame_time + 0.0334},"
                                f"setpts=PTS-STARTPTS,"
                                f"select='eq(n,0)',"
                                f"scale=576:1024,"
                                f"setsar=1,"
                                f"zoompan="
                                f"z='if(lte(on,29),"
                                f"1+0.15*on/29,"
                                f"1.15-0.15*(on-29)/29)':"
                                f"d=60:"
                                f"x='iw/2-(iw/zoom/2)':"
                                f"y='ih/2-(ih/zoom/2)':"
                                f"s=576x1024:"
                                f"fps=30"
                                f"[{freeze_label}]"
                            )


                            filter_parts.append(
                                freeze_filter
                            )


                            labels.append(
                                f"[{freeze_label}]"
                            )


                    if labels:

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

                        filter_parts.append(
                            "[0:v]"
                            "scale=576:1024,"
                            "setsar=1,"
                            "format=yuv420p"
                            "[basevideo]"
                        )

                else:

                    filter_parts.append(
                        "[0:v]"
                        "scale=576:1024,"
                        "setsar=1,"
                        "format=yuv420p"
                        "[basevideo]"
                    )


                # =================================================
                # SUBTITLE OVERLAY
                # =================================================

                subtitle_filter = (
                    "[basevideo]"
                    f"ass=filename='{ass_filter_file}'"
                    f":fontsdir='{ass_fonts_dir}'"
                    "[vout]"
                )


                filter_parts.append(
                    subtitle_filter
                )


                # =================================================
                # PART VOICEOVER
                # =================================================

                # -----------------------------------------------
                # We take the correct section of the global
                # voiceover for this video part.
                # -----------------------------------------------

                voice_local_start = max(
                    0,
                    part_start
                )


                voice_local_duration = max(
                    0,
                    min(
                        part_end,
                        voice_duration
                    )
                    - voice_local_start
                )


                # =================================================
                # PART OUTPUT FILE
                # =================================================

                part_output = os.path.join(
                    export_dir,
                    f"part_{part_index:03d}.mp4"
                )


                # =================================================
                # BUILD FILTER COMPLEX
                # =================================================

                filter_complex = ";".join(
                    filter_parts
                )


                # =================================================
                # PART AUDIO FILTER
                # =================================================

                if voice_local_duration > 0:

                    audio_filter = (
                        f"[1:a:0]"
                        f"atrim="
                        f"start={voice_local_start}:"
                        f"end={min(part_end, voice_duration)},"
                        f"asetpts=PTS-STARTPTS"
                        f"[aout]"
                    )

                    filter_complex += (
                        ";"
                        + audio_filter
                    )

                    final_audio_label = (
                        "[aout]"
                    )

                else:

                    audio_filter = (
                        "anullsrc="
                        "channel_layout=stereo:"
                        "sample_rate=48000,"
                        f"atrim=duration={part_video_duration}"
                        "[aout]"
                    )

                    filter_complex += (
                        ";"
                        + audio_filter
                    )

                    final_audio_label = (
                        "[aout]"
                    )


                # =================================================
                # FREEZE DURATION CALCULATION
                # =================================================

                if freeze_enabled:

                    local_freeze_count = max(
                        0,
                        local_segment_count - 1
                    )

                else:

                    local_freeze_count = 0


                part_base_duration = (
                    part_video_duration
                    +
                    (
                        local_freeze_count
                        * freeze_time
                    )
                )


                # =================================================
                # AUDIO DURATION FOR PART
                # =================================================

                part_voice_duration = (
                    voice_local_duration
                )


                part_target_duration = max(
                    part_base_duration,
                    part_voice_duration
                )


                part_extra_duration = max(
                    0,
                    part_voice_duration
                    - part_base_duration
                )


                # =================================================
                # EXTEND FINAL FRAME IF VOICE IS LONGER
                # =================================================

                if part_extra_duration > 0:

                    filter_parts.append(
                        "[vout]"
                        f"tpad="
                        f"stop_mode=clone:"
                        f"stop_duration="
                        f"{part_extra_duration:.3f}:"
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
                # REBUILD FILTER COMPLEX
                # =================================================

                # The tpad filter may have been added after the
                # previous filter_complex string was created.
                # Therefore rebuild everything from filter_parts.
                #
                # First remove the previous audio filter if needed.
                # =================================================

                video_filter_complex = ";".join(
                    filter_parts
                )


                # =================================================
                # AUDIO PADDING
                # =================================================

                audio_pad_duration = max(
                    0,
                    part_base_duration
                    - part_voice_duration
                )


                if audio_pad_duration > 0:

                    audio_filter = (
                        f"[1:a:0]"
                        f"atrim="
                        f"start={voice_local_start}:"
                        f"end={min(part_end, voice_duration)},"
                        f"asetpts=PTS-STARTPTS,"
                        f"apad="
                        f"pad_dur={audio_pad_duration:.3f}"
                        "[aout]"
                    )

                else:

                    audio_filter = (
                        f"[1:a:0]"
                        f"atrim="
                        f"start={voice_local_start}:"
                        f"end={min(part_end, voice_duration)},"
                        f"asetpts=PTS-STARTPTS"
                        "[aout]"
                    )


                # =================================================
                # FINAL PART FILTER COMPLEX
                # =================================================

                filter_complex = (
                    video_filter_complex
                    + ";"
                    + audio_filter
                )


                # =================================================
                # FFMPEG PART COMMAND
                # =================================================

                command = [
                    "ffmpeg",
                    "-y",

                    "-ss",
                    f"{part_start:.3f}",

                    "-i",
                    original_video,

                    "-i",
                    voice_file,

                    "-filter_complex",
                    filter_complex,

                    "-map",
                    final_video_label,

                    "-map",
                    "[aout]",

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
                    f"{part_target_duration:.3f}",

                    part_output
                ]


                # =================================================
                # EXPORT PART
                # =================================================

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )


                if result.returncode != 0:

                    st.error(
                        f"❌ Part "
                        f"{part_index + 1}"
                        f" Export Failed"
                    )

                    st.code(
                        result.stderr[-5000:],
                        language="text"
                    )

                    raise RuntimeError(
                        f"Part {part_index + 1} export failed."
                    )


                part_files.append(
                    part_output
                )


                st.success(
                    f"✅ Part "
                    f"{part_index + 1}/{part_count}"
                    f" completed."
                )


            # =====================================================
            # COMBINE PARTS
            # =====================================================

            if len(part_files) == 1:

                output_video = part_files[0]

            else:

                st.info(
                    "🔗 Combining exported parts..."
                )


                concat_file = os.path.join(
                    export_dir,
                    "concat.txt"
                )


                with open(
                    concat_file,
                    "w",
                    encoding="utf-8"
                ) as f:

                    for part_file in part_files:

                        safe_path = (
                            part_file
                            .replace("\\", "/")
                            .replace("'", "'\\''")
                        )

                        f.write(
                            f"file '{safe_path}'\n"
                        )


                output_video = os.path.join(
                    tempfile.gettempdir(),
                    "final_movie_recap.mp4"
                )


                concat_command = [
                    "ffmpeg",
                    "-y",

                    "-f",
                    "concat",

                    "-safe",
                    "0",

                    "-i",
                    concat_file,

                    "-c",
                    "copy",

                    output_video
                ]


                concat_result = subprocess.run(
                    concat_command,
                    capture_output=True,
                    text=True
                )


                if concat_result.returncode != 0:

                    st.error(
                        "❌ Could not combine exported parts."
                    )

                    st.code(
                        concat_result.stderr[-5000:],
                        language="text"
                    )

                    raise RuntimeError(
                        "Final video combination failed."
                    )


                st.success(
                    f"🔗 {len(part_files)} parts "
                    f"combined successfully."
                )


            # =================================================
            # FINAL DURATION
            # =================================================

            final_duration = get_audio_duration(
                output_video
            )


            # =================================================
            # FINAL RESULT
            # =================================================

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


            st.info(
                f"🔤 Subtitle Font: "
                f"{subtitle_font}"
            )


            st.info(
                "🟡 Subtitle Color: Yellow"
            )


            st.info(
                f"✂️ Export Parts: "
                f"{len(part_files)}"
            )


            st.info(
                "🤖 Gemini API: "
                "No additional Gemini calls during export."
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
                
