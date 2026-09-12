import streamlit as st
import os
from google import genai
import cv2
import tempfile
import subprocess
import math
import re
import asyncio
import shutil

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Movie Recap AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recap AI")
st.write("Upload a movie and analyze video information.")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_audio_duration(media_file):

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


def ass_time(seconds):

    seconds = max(0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{centiseconds:02d}"
    )


def split_myanmar_text(text, max_chars=65):

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

            chunks.append(sentence)
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
                    chunks.append(current)

                current = word

        if current:
            chunks.append(current)

    return chunks


def wrap_myanmar(text, max_chars=24):

    words = str(text).split()

    if not words:
        return ""

    lines = []
    current = ""

    for word in words:

        test = (
            word
            if not current
            else current + " " + word
        )

        if len(test) <= max_chars:

            current = test

        else:

            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    if len(lines) > 3:
        lines = lines[:3]

    return "\\N".join(lines)


def extract_preview_frame(
    video_path,
    time_seconds,
    width=576,
    height=1024
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        float(time_seconds) * 1000
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame)

    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return image


# =========================================================
# BLUR DRAWING JSON
# =========================================================

def make_blur_drawing(
    x,
    y,
    width,
    height
):

    return {
        "version": "7.0.0",
        "objects": [
            {
                "type": "Rect",

                "version": "7.0.0",

                "left": float(x),
                "top": float(y),

                "width": float(width),
                "height": float(height),

                "scaleX": 1,
                "scaleY": 1,

                "angle": 0,

                "originX": "left",
                "originY": "top",

                "fill": "rgba(255,0,0,0.20)",

                "stroke": "#FF0000",

                "strokeWidth": 4,

                "selectable": True,
                "evented": True,

                "hasControls": True,
                "hasBorders": True,

                "lockRotation": True
            }
        ]
    }


# =========================================================
# NORMALIZE BLUR BOX
# =========================================================

def normalize_blur_box(
    x,
    y,
    width,
    height
):

    x = int(round(x))
    y = int(round(y))
    width = int(round(width))
    height = int(round(height))

    x = max(
        0,
        min(575, x)
    )

    y = max(
        0,
        min(1023, y)
    )

    width = max(
        10,
        min(
            576 - x,
            width
        )
    )

    height = max(
        10,
        min(
            1024 - y,
            height
        )
    )

    return (
        x,
        y,
        width,
        height
    )


# =========================================================
# CACHED WHISPER MODEL
# =========================================================

@st.cache_resource
def load_whisper_model(model_name="tiny"):

    import whisper

    return whisper.load_model(model_name)


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

        temp_file.write(video_bytes)

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

        st.divider()

        st.subheader(
            "🎥 Video Preview"
        )

        st.video(video_bytes)

        cap.release()

    st.divider()


# =========================================================
# MOVIE TRANSCRIPT
# =========================================================

st.subheader(
    "🎤 Movie Transcript"
)


if uploaded_file is not None:

    whisper_model = st.selectbox(
        "🧠 Whisper Model",
        ["tiny", "base"],
        index=0,
        help=(
            "Tiny uses much less CPU/RAM. "
            "Base may provide better English transcription."
        )
    )

    if st.button(
        "📝 Generate Transcript"
    ):

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
    ):

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

            st.success(
                f"✅ {len(segments)} scenes analyzed!"
            )

            for i, segment in enumerate(
                segments
            ):

                start = segment.get(
                    "start",
                    0
                )

                end = segment.get(
                    "end",
                    0
                )

                text = segment.get(
                    "text",
                    ""
                ).strip()

                if text:

                    st.markdown(
                        f"### Scene {i + 1} "
                        f"({start:.1f}s → {end:.1f}s)"
                    )

                    st.write(text)


st.divider()


# =========================================================
# AI SCENE ANALYSIS
# =========================================================

st.subheader(
    "🤖 AI Scene Analysis"
)


if "transcript" in st.session_state:

    if st.button(
        "🤖 Analyze with Gemini"
    ):

        try:

            api_key = st.secrets[
                "GEMINI_API_KEY"
            ]

            client = genai.Client(
                api_key=api_key
            )

            transcript_text = (
                st.session_state[
                    "transcript"
                ]
            )

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

            response = (
                client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
            )

            st.session_state[
                "ai_scene_analysis"
            ] = response.text

            st.success(
                "✅ Gemini Scene Analysis completed!"
            )

        except Exception as e:

            st.error(
                f"❌ Gemini Analysis failed: {e}"
            )


if "ai_scene_analysis" in st.session_state:

    st.text_area(
        "🤖 AI Scene Analysis Result",
        st.session_state[
            "ai_scene_analysis"
        ],
        height=500
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
    ):

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
    ):

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

            response = (
                client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
            )

            st.session_state[
                "myanmar_recap"
            ] = response.text

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
# MYANMAR FEMALE VOICEOVER
# =========================================================

st.subheader(
    "🎙️ Myanmar Female Voiceover"
)


if "myanmar_recap" in st.session_state:

    voice_speed = st.selectbox(
        "🎙️ Voice Speed",
        [1.0, 1.1, 1.2],
        index=0,
        key="voice_speed_select"
    )

    st.caption(
        "CPU optimized TTS pipeline."
    )

    if st.button(
        "🎙️ Generate Myanmar Voiceover"
    ):

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
                max_chars=65
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

            cumulative_raw_time = 0.0

            progress = st.progress(0)

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
                        chunk,
                        "my-MM-NilarNeural"
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

                cumulative_raw_time += duration

                progress.progress(
                    (index + 1)
                    / len(raw_files)
                )

            concat_file = os.path.join(
                work_dir,
                "concat_mp3.txt"
            )

            with open(
                concat_file,
                "w",
                encoding="utf-8"
            ) as f:

                for raw_file in raw_files:

                    safe_path = (
                        os.path.abspath(
                            raw_file
                        ).replace(
                            "'",
                            "'\\''"
                        )
                    )

                    f.write(
                        f"file '{safe_path}'\n"
                    )

            combined_mp3 = os.path.join(
                work_dir,
                "combined.mp3"
            )

            concat_result = subprocess.run(
                [
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
                    combined_mp3
                ],
                capture_output=True,
                text=True
            )

            if concat_result.returncode != 0:

                raise RuntimeError(
                    concat_result.stderr[-3000:]
                )

            final_voice_file = os.path.join(
                work_dir,
                "myanmar_voiceover.mp3"
            )

            if float(voice_speed) == 1.0:

                shutil.copyfile(
                    combined_mp3,
                    final_voice_file
                )

            else:

                speed_result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        combined_mp3,
                        "-filter:a",
                        f"atempo={float(voice_speed)}",
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        "128k",
                        final_voice_file
                    ],
                    capture_output=True,
                    text=True
                )

                if speed_result.returncode != 0:

                    raise RuntimeError(
                        speed_result.stderr[-3000:]
                    )

            voice_duration = get_audio_duration(
                final_voice_file
            )

            predicted_duration = (
                cumulative_raw_time
                / float(voice_speed)
            )

            if predicted_duration > 0:

                timing_scale = (
                    voice_duration
                    / predicted_duration
                )

            else:

                timing_scale = 1.0

            timing_scale = max(
                0.98,
                min(
                    1.02,
                    timing_scale
                )
            )

            subtitle_data = []

            raw_cursor = 0.0

            for index, chunk in enumerate(
                chunks
            ):

                raw_start = raw_cursor

                raw_end = (
                    raw_cursor
                    + raw_durations[index]
                )

                start_time = (
                    raw_start
                    / float(voice_speed)
                )

                end_time = (
                    raw_end
                    / float(voice_speed)
                )

                start_time *= timing_scale
                end_time *= timing_scale

                if start_time < voice_duration:

                    end_time = min(
                        end_time,
                        voice_duration
                    )

                    if end_time > start_time:

                        subtitle_data.append({
                            "start": start_time,
                            "end": end_time,
                            "text": chunk
                        })

                raw_cursor = raw_end

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
            ] = "tts_segments"

            st.session_state[
                "voice_chunks"
            ] = chunks

            st.session_state[
                "voice_speed_value"
            ] = float(voice_speed)

            st.success(
                "✅ Myanmar Female Voiceover generated!"
            )

            st.success(
                f"✅ {len(subtitle_data)} subtitles "
                f"were synced to the generated voice."
            )

            st.info(
                f"🎙️ Voiceover Duration: "
                f"{voice_duration:.2f} seconds"
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
        "ℹ️ Whisper is NOT used for Myanmar subtitle timing."
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
            f"{start_time} --> {end_time}\n"
            f"{text}\n"
        )

    srt_content = "\n".join(
        srt_lines
    )

    st.download_button(
        "📥 Download Myanmar Subtitle (.srt)",
        data=srt_content.encode("utf-8"),
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

    subtitle_duration = (
        st.session_state.get(
            "voice_duration",
            0
        )
    )

    fixed_subtitles = []

    for item in st.session_state[
        "subtitle_data"
    ]:

        start = max(
            0,
            float(item["start"])
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

        fixed_subtitles.append({
            "start": start,
            "end": end,
            "text": text
        })

    st.session_state[
        "subtitle_data"
    ] = fixed_subtitles

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


freeze_enabled = st.checkbox(
    "Enable Freeze Frame + Zoom",
    value=True,
    key="freeze_enabled"
)


freeze_interval = st.number_input(
    "⏱️ Freeze Every",
    min_value=5.0,
    max_value=60.0,
    value=10.0,
    step=1.0,
    key="freeze_interval"
)


freeze_duration = st.number_input(
    "🧊 Freeze Duration",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    key="freeze_duration"
)


st.caption(
    "Every interval → Freeze + Zoom In → Zoom Out"
)


st.divider()


# =========================================================
# ORIGINAL SUBTITLE BLUR
# =========================================================

st.subheader(
    "🟦 Original Subtitle Blur"
)


blur_enabled = st.checkbox(
    "🟦 Blur Mode ON / OFF",
    value=False,
    key="blur_enabled"
)


# =========================================================
# BLUR SESSION STATE
# =========================================================

if "blur_x" not in st.session_state:
    st.session_state["blur_x"] = 20

if "blur_y" not in st.session_state:
    st.session_state["blur_y"] = 860

if "blur_width" not in st.session_state:
    st.session_state["blur_width"] = 536

if "blur_height" not in st.session_state:
    st.session_state["blur_height"] = 100

if "blur_strength" not in st.session_state:
    st.session_state["blur_strength"] = 15

if "blur_canvas_reset" not in st.session_state:
    st.session_state["blur_canvas_reset"] = 0


# =========================================================
# BLUR MODE
# =========================================================

if blur_enabled:

    st.success(
        "🟦 Blur Mode ON"
    )

    st.info(
        "📱 အနီရောင် Box ကို ဖိဆွဲပြီး ရွှေ့ပါ။ "
        "Corner/side handles ကို ဆွဲပြီး "
        "အရွယ်အစားချိန်ပါ။"
    )

    blur_video_path = st.session_state.get(
        "video_path",
        None
    )

    blur_duration = st.session_state.get(
        "video_duration",
        0
    )

    if (
        blur_video_path
        and os.path.exists(
            blur_video_path
        )
    ):

        max_preview_time = max(
            0.0,
            float(blur_duration) - 0.1
        )

        default_preview_time = min(
            1.0,
            max_preview_time
        )

        blur_preview_time = st.slider(
            "🎞️ Preview Time",
            min_value=0.0,
            max_value=float(
                max_preview_time
            ),
            value=float(
                default_preview_time
            ),
            step=0.5,
            key="blur_preview_time"
        )

        preview_frame = extract_preview_frame(
            blur_video_path,
            blur_preview_time,
            width=576,
            height=1024
        )

        if preview_frame is not None:

            st.markdown(
                "### 🎯 Blur Preview"
            )

            # ---------------------------------------------
            # CURRENT BOX
            # ---------------------------------------------

            current_x = int(
                st.session_state.get(
                    "blur_x",
                    20
                )
            )

            current_y = int(
                st.session_state.get(
                    "blur_y",
                    860
                )
            )

            current_w = int(
                st.session_state.get(
                    "blur_width",
                    536
                )
            )

            current_h = int(
                st.session_state.get(
                    "blur_height",
                    100
                )
            )

            # ---------------------------------------------
            # INITIAL BOX
            # ---------------------------------------------

            initial_drawing = make_blur_drawing(
                current_x,
                current_y,
                current_w,
                current_h
            )

            # ---------------------------------------------
            # STABLE KEY
            # ---------------------------------------------

            canvas_key = (
                "blur_canvas_"
                + str(
                    st.session_state[
                        "blur_canvas_reset"
                    ]
                )
            )

            # ---------------------------------------------
            # CANVAS
            # ---------------------------------------------

            canvas_result = st_canvas(
                fill_color="rgba(255,0,0,0.20)",
                stroke_width=4,
                stroke_color="#FF0000",
                background_image=preview_frame,
                background_image_fit="stretch",
                update_streamlit=True,
                height=1024,
                width=576,
                drawing_mode="rect",
                initial_drawing=initial_drawing,
                display_toolbar=True,
                max_display_height=700,
                key=canvas_key
            )

            # ---------------------------------------------
            # READ OBJECT
            # ---------------------------------------------

            if (
                canvas_result is not None
                and canvas_result.json_data is not None
            ):

                try:

                    objects = (
                        canvas_result.json_data.get(
                            "objects",
                            []
                        )
                    )

                    rect_objects = [
                        obj
                        for obj in objects
                        if str(
                            obj.get(
                                "type",
                                ""
                            )
                        ).lower()
                        == "rect"
                    ]

                    if rect_objects:

                        # Use last rectangle
                        obj = rect_objects[-1]

                        raw_x = float(
                            obj.get(
                                "left",
                                current_x
                            )
                        )

                        raw_y = float(
                            obj.get(
                                "top",
                                current_y
                            )
                        )

                        raw_w = float(
                            obj.get(
                                "width",
                                current_w
                            )
                        )

                        raw_h = float(
                            obj.get(
                                "height",
                                current_h
                            )
                        )

                        scale_x = float(
                            obj.get(
                                "scaleX",
                                1
                            )
                        )

                        scale_y = float(
                            obj.get(
                                "scaleY",
                                1
                            )
                        )

                        # ---------------------------------
                        # Fabric scaling
                        # ---------------------------------

                        final_w = (
                            raw_w
                            * scale_x
                        )

                        final_h = (
                            raw_h
                            * scale_y
                        )

                        final_x = raw_x
                        final_y = raw_y

                        # ---------------------------------
                        # Origin correction
                        # ---------------------------------

                        origin_x = obj.get(
                            "originX",
                            "left"
                        )

                        origin_y = obj.get(
                            "originY",
                            "top"
                        )

                        if origin_x == "center":

                            final_x -= (
                                final_w / 2
                            )

                        if origin_y == "center":

                            final_y -= (
                                final_h / 2
                            )

                        (
                            final_x,
                            final_y,
                            final_w,
                            final_h
                        ) = normalize_blur_box(
                            final_x,
                            final_y,
                            final_w,
                            final_h
                        )

                        # ---------------------------------
                        # Save coordinates
                        # ---------------------------------

                        st.session_state[
                            "blur_x"
                        ] = final_x

                        st.session_state[
                            "blur_y"
                        ] = final_y

                        st.session_state[
                            "blur_width"
                        ] = final_w

                        st.session_state[
                            "blur_height"
                        ] = final_h

                except Exception as e:

                    st.warning(
                        f"⚠️ Blur box reading issue: {e}"
                    )

            # ---------------------------------------------
            # RESET
            # ---------------------------------------------

            if st.button(
                "🔄 Reset Blur Box",
                key="reset_blur_box"
            ):

                st.session_state[
                    "blur_x"
                ] = 20

                st.session_state[
                    "blur_y"
                ] = 860

                st.session_state[
                    "blur_width"
                ] = 536

                st.session_state[
                    "blur_height"
                ] = 100

                st.session_state[
                    "blur_canvas_reset"
                ] += 1

                st.rerun()

            # ---------------------------------------------
            # BLUR STRENGTH
            # ---------------------------------------------

            blur_strength = st.slider(
                "🌫️ Blur Strength",
                min_value=1,
                max_value=30,
                value=int(
                    st.session_state[
                        "blur_strength"
                    ]
                ),
                step=1,
                key="blur_strength_slider"
            )

            st.session_state[
                "blur_strength"
            ] = int(
                blur_strength
            )

            # ---------------------------------------------
            # STATUS
            # ---------------------------------------------

            st.success(
                "✅ Blur Box Selected"
            )

            st.info(
                f"📍 Position: "
                f"X={st.session_state['blur_x']}, "
                f"Y={st.session_state['blur_y']}"
            )

            st.info(
                f"🟦 Size: "
                f"{st.session_state['blur_width']} × "
                f"{st.session_state['blur_height']} px"
            )

            st.info(
                f"🌫️ Blur Strength: "
                f"{blur_strength}"
            )

        else:

            st.error(
                "❌ Preview frame မဖတ်နိုင်ပါ။"
            )

    else:

        st.warning(
            "⚠️ Video upload အရင်လုပ်ပါ။"
        )

else:

    st.caption(
        "⬜ Blur Mode OFF — "
        "Original subtitles remain visible."
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
    and "subtitle_data" in st.session_state
    and "uploaded_file" in st.session_state
):

    if st.button(
        "🎬 Create Final Recap Video"
    ):

        try:

            # =============================================
            # SAVE VIDEO
            # =============================================

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

            # =============================================
            # VIDEO DURATION
            # =============================================

            video_duration = get_audio_duration(
                original_video
            )

            # =============================================
            # VOICE DURATION
            # =============================================

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

            # =============================================
            # FREEZE SETTINGS
            # =============================================

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

            # =============================================
            # ASS
            # =============================================

            ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 576
PlayResY: 1024
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Myanmar,Noto Sans Myanmar,28,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,0,0,0,0,100,100,0,0,1,2,1,2,40,40,55,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

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
                    24
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

            # =============================================
            # FILTER
            # =============================================

            filter_parts = []
            labels = []

            # =============================================
            # FREEZE + ZOOM
            # =============================================

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
                        f"normal{i}"
                    )

                    normal_filter = (
                        f"[0:v]"
                        f"trim=start={start}:end={end},"
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

                    if end < video_duration:

                        freeze_label = (
                            f"freeze{i}"
                        )

                        frame_time = max(
                            start,
                            end - 0.10
                        )

                        freeze_frames = max(
                            1,
                            int(
                                round(
                                    freeze_time * 30
                                )
                            )
                        )

                        zoom_frames = max(
                            1,
                            freeze_frames // 2
                        )

                        zoom_filter = (
                            f"[0:v]"
                            f"trim=start={frame_time}:"
                            f"end={frame_time + 0.0334},"
                            f"setpts=PTS-STARTPTS,"
                            f"select='eq(n,0)',"
                            f"scale=576:1024,"
                            f"setsar=1,"
                            f"zoompan="
                            f"z='if(lte(on,"
                            f"{zoom_frames - 1}),"
                            f"1+0.15*on/"
                            f"{max(1, zoom_frames - 1)},"
                            f"1.15-0.15*(on-{zoom_frames})/"
                            f"{max(1, zoom_frames - 1)})':"
                            f"d={freeze_frames}:"
                            f"x='iw/2-(iw/zoom/2)':"
                            f"y='ih/2-(ih/zoom/2)':"
                            f"s=576x1024:"
                            f"fps=30"
                            f"[{freeze_label}]"
                        )

                        filter_parts.append(
                            zoom_filter
                        )

                        labels.append(
                            f"[{freeze_label}]"
                        )

                concat_inputs = "".join(
                    labels
                )

                concat_filter = (
                    f"{concat_inputs}"
                    f"concat=n={len(labels)}:"
                    f"v=1:a=0:"
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

            # =============================================
            # BLUR
            # =============================================

            video_for_subtitle = (
                "[basevideo]"
            )

            if blur_enabled:

                bx = int(
                    st.session_state.get(
                        "blur_x",
                        20
                    )
                )

                by = int(
                    st.session_state.get(
                        "blur_y",
                        860
                    )
                )

                bw = int(
                    st.session_state.get(
                        "blur_width",
                        536
                    )
                )

                bh = int(
                    st.session_state.get(
                        "blur_height",
                        100
                    )
                )

                br = int(
                    st.session_state.get(
                        "blur_strength",
                        15
                    )
                )

                (
                    bx,
                    by,
                    bw,
                    bh
                ) = normalize_blur_box(
                    bx,
                    by,
                    bw,
                    bh
                )

                br = max(
                    1,
                    min(
                        30,
                        br
                    )
                )

                blur_filter = (
                    f"[basevideo]"
                    f"split=2"
                    f"[blur_main]"
                    f"[blur_source];"

                    f"[blur_source]"
                    f"crop={bw}:{bh}:{bx}:{by},"
                    f"boxblur="
                    f"luma_radius={br}:"
                    f"luma_power=2"
                    f"[blur_region];"

                    f"[blur_main]"
                    f"[blur_region]"
                    f"overlay={bx}:{by}:"
                    f"shortest=1"
                    f"[blurredvideo]"
                )

                filter_parts.append(
                    blur_filter
                )

                video_for_subtitle = (
                    "[blurredvideo]"
                )

            # =============================================
            # MYANMAR SUBTITLE
            # =============================================

            filter_parts.append(
                f"{video_for_subtitle}"
                f"ass={ass_file}"
                f"[vout]"
            )

            # =============================================
            # DURATION
            # =============================================

            if freeze_enabled:

                actual_freeze_count = max(
                    0,
                    segment_count - 1
                )

            else:

                actual_freeze_count = 0

            base_video_duration = (
                video_duration
                + (
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

            if blur_enabled:

                st.info(
                    "🟦 Original Subtitle Blur: ON"
                )

                st.caption(
                    f"X={bx}, Y={by}, "
                    f"W={bw}, H={bh}, "
                    f"Strength={br}"
                )

            else:

                st.info(
                    "⬜ Original Subtitle Blur: OFF"
                )

            # =============================================
            # EXTEND
            # =============================================

            if extra_duration > 0:

                filter_parts.append(
                    "[vout]"
                    f"tpad="
                    f"stop_mode=clone:"
                    f"stop_duration={extra_duration:.3f},"
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

            # =============================================
            # FILTER COMPLEX
            # =============================================

            filter_complex = ";".join(
                filter_parts
            )

            # =============================================
            # OUTPUT
            # =============================================

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

            # =============================================
            # EXPORT
            # =============================================

            st.info(
                "⏳ Creating final video..."
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
                    "✅ Final Recap Video Created Successfully!"
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
