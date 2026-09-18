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
    """

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if not text:
        return ""

    if len(text) <= max_chars:

        return text

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
# VIDEO SPLIT SETTINGS
# =========================================================

SPLIT_DURATION = 60.0


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
                f"{split_count} × 60-second "
                f"processing parts."
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

            # =================================================
            # EXTRACT AUDIO FOR WHISPER
            #
            # Only this part is changed.
            #
            # The video is converted to a clean:
            # - Mono
            # - 16 kHz
            # - PCM WAV
            #
            # Then Whisper transcribes the WAV file.
            # =================================================

            video_path_for_whisper = (
                st.session_state[
                    "video_path"
                ]
            )

            whisper_work_dir = tempfile.mkdtemp(
                prefix="movie_recap_whisper_"
            )

            whisper_audio_path = os.path.join(
                whisper_work_dir,
                "whisper_audio.wav"
            )

            audio_extract_result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path_for_whisper,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    whisper_audio_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if (
                audio_extract_result.returncode != 0
                or
                not os.path.isfile(
                    whisper_audio_path
                )
            ):

                raise RuntimeError(
                    "Could not extract audio from the video.\n\n"
                    +
                    audio_extract_result.stderr[-3000:]
                )

            # =================================================
            # CHECK EXTRACTED AUDIO
            # =================================================

            audio_file_size = os.path.getsize(
                whisper_audio_path
            )

            if audio_file_size < 1000:

                raise RuntimeError(
                    "The extracted audio file is empty "
                    "or too small to transcribe."
                )

            whisper_audio_duration = (
                get_audio_duration(
                    whisper_audio_path
                )
            )

            if whisper_audio_duration <= 0.05:

                raise RuntimeError(
                    "No usable audio track was found "
                    "in this video."
                )

            st.info(
                f"🎧 Whisper Audio: "
                f"{whisper_audio_duration:.2f} sec"
            )

            # =================================================
            # WHISPER TRANSCRIPTION
            # =================================================

            result = model.transcribe(
                whisper_audio_path,
                language="en",
                fp16=False
            )

            if not result:

                raise RuntimeError(
                    "Whisper returned no transcription result."
                )

            transcript_text = str(
                result.get(
                    "text",
                    ""
                )
            ).strip()

            if not transcript_text:

                raise RuntimeError(
                    "Whisper completed but returned "
                    "an empty transcript."
                )

            st.session_state[
                "transcript_result"
            ] = result

            st.session_state[
                "transcript"
            ] = transcript_text

            st.session_state[
                "whisper_audio_path"
            ] = whisper_audio_path

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

    st.caption(
        "✂️ Videos longer than 1 minute are automatically "
        "processed in 60-second timeline parts."
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
            "Video, Voiceover and Subtitle timing "
            "use the same final timeline."
        )

    else:

        st.info(
            "🎬 Split Mode OFF — "
            "Video will be exported as one timeline."
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
            # READ DURATIONS
            # =================================================

            video_duration = get_audio_duration(
                original_video
            )

            voice_duration = get_audio_duration(
                voice_file
            )

            st.session_state[
                "voice_duration"
            ] = voice_duration

            st.info(
                f"🎬 Source Video: "
                f"{video_duration:.2f} sec"
            )

            st.info(
                f"🎙️ Voiceover: "
                f"{voice_duration:.2f} sec"
            )


            # =================================================
            # FREEZE SETTINGS
            #
            # IMPORTANT:
            # Freeze does NOT add extra time to the final
            # timeline.
            #
            # The freeze replaces that portion of the
            # source video timeline.
            # =================================================

            if freeze_enabled:

                interval = float(
                    freeze_interval
                )

                freeze_time = min(
                    float(freeze_duration),
                    interval
                )

            else:

                interval = (
                    video_duration + 1
                )

                freeze_time = 0.0


            # =================================================
            # SUBTITLE SETTINGS
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
            # EXPORT DIRECTORY
            # =================================================

            export_dir = tempfile.mkdtemp(
                prefix="movie_recap_export_"
            )


            # =================================================
            # IMPORTANT TIMELINE
            #
            # Freeze no longer increases video duration.
            #
            # Therefore:
            #
            # Final Video Timeline =
            # max(Source Video, Voiceover)
            #
            # This keeps Video / Voice / Subtitle on
            # the same timeline.
            # =================================================

            total_target_duration = max(
                video_duration,
                voice_duration
            )


            # =================================================
            # PART COUNT
            # =================================================

            if (
                split_mode
                and
                total_target_duration > split_duration
            ):

                part_count = int(
                    math.ceil(
                        total_target_duration
                        / split_duration
                    )
                )

            else:

                part_count = 1


            st.write(
                f"✂️ Export Parts: "
                f"{part_count}"
            )

            st.write(
                f"🎬 Final Timeline: "
                f"{total_target_duration:.2f} sec"
            )


            part_files = []


            # =================================================
            # CREATE EACH PART
            # =================================================

            for part_index in range(
                part_count
            ):

                # ---------------------------------------------
                # PART TIMELINE
                # ---------------------------------------------

                if part_count > 1:

                    part_start = (
                        part_index
                        * split_duration
                    )

                    part_end = min(
                        part_start
                        + split_duration,
                        total_target_duration
                    )

                else:

                    part_start = 0.0

                    part_end = (
                        total_target_duration
                    )


                part_duration = (
                    part_end
                    - part_start
                )


                st.info(
                    f"⏳ Part "
                    f"{part_index + 1}/{part_count} "
                    f"• "
                    f"{part_start:.2f}s → "
                    f"{part_end:.2f}s"
                )


                # =================================================
                # SOURCE VIDEO AVAILABLE IN THIS PART
                # =================================================

                source_part_start = max(
                    0.0,
                    part_start
                )

                source_part_end = min(
                    video_duration,
                    part_end
                )


                has_source_video = (
                    source_part_start
                    <
                    source_part_end
                    - 0.0001
                )


                # =================================================
                # ASS SUBTITLE
                #
                # IMPORTANT:
                # Subtitle timing uses FINAL TIMELINE.
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


                for item in st.session_state[
                    "subtitle_data"
                ]:

                    global_start = max(
                        0.0,
                        float(
                            item["start"]
                        )
                    )

                    global_end = max(
                        global_start + 0.05,
                        float(
                            item["end"]
                        )
                    )


                    if global_end <= part_start:

                        continue

                    if global_start >= part_end:

                        continue


                    clipped_start = max(
                        global_start,
                        part_start
                    )

                    clipped_end = min(
                        global_end,
                        part_end
                    )


                    local_start = (
                        clipped_start
                        -
                        part_start
                    )

                    local_end = (
                        clipped_end
                        -
                        part_start
                    )


                    if local_end <= local_start:

                        continue


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
                # SAVE ASS
                # =================================================

                ass_file = os.path.join(
                    export_dir,
                    f"subtitle_part_{part_index:03d}.ass"
                )


                with open(
                    ass_file,
                    "w",
                    encoding="utf-8-sig"
                ) as f:

                    f.write(
                        ass_content
                    )


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
                # VIDEO FILTER
                # =================================================

                filter_parts = []

                video_labels = []


                # =================================================
                # SOURCE VIDEO EXISTS
                # =================================================

                if has_source_video:

                    cursor = (
                        source_part_start
                    )


                    if freeze_enabled:

                        first_index = int(
                            math.ceil(
                                (
                                    source_part_start
                                    -
                                    0.000001
                                )
                                /
                                interval
                            )
                        )

                        if first_index < 1:

                            first_index = 1

                        next_freeze = (
                            first_index
                            * interval
                        )

                    else:

                        next_freeze = (
                            video_duration
                            + 1
                        )


                    piece_index = 0


                    while (
                        cursor
                        <
                        source_part_end
                        -
                        0.0001
                    ):

                        if (
                            not freeze_enabled
                            or
                            next_freeze
                            >=
                            source_part_end
                            -
                            0.0001
                        ):

                            normal_start = (
                                cursor
                            )

                            normal_end = (
                                source_part_end
                            )


                            if (
                                normal_end
                                >
                                normal_start
                                +
                                0.0001
                            ):

                                label = (
                                    f"normal_"
                                    f"{part_index}_"
                                    f"{piece_index}"
                                )

                                filter_parts.append(
                                    f"[0:v]"
                                    f"trim="
                                    f"start={normal_start:.6f}:"
                                    f"end={normal_end:.6f},"
                                    f"setpts=PTS-STARTPTS,"
                                    f"scale=576:1024,"
                                    f"setsar=1,"
                                    f"format=yuv420p"
                                    f"[{label}]"
                                )

                                video_labels.append(
                                    f"[{label}]"
                                )

                                piece_index += 1


                            cursor = (
                                source_part_end
                            )

                            break


                        normal_start = (
                            cursor
                        )

                        normal_end = min(
                            next_freeze,
                            source_part_end
                        )


                        if (
                            normal_end
                            >
                            normal_start
                            +
                            0.0001
                        ):

                            label = (
                                f"normal_"
                                f"{part_index}_"
                                f"{piece_index}"
                            )


                            filter_parts.append(
                                f"[0:v]"
                                f"trim="
                                f"start={normal_start:.6f}:"
                                f"end={normal_end:.6f},"
                                f"setpts=PTS-STARTPTS,"
                                f"scale=576:1024,"
                                f"setsar=1,"
                                f"format=yuv420p"
                                f"[{label}]"
                            )


                            video_labels.append(
                                f"[{label}]"
                            )


                            piece_index += 1


                        freeze_start = (
                            next_freeze
                        )

                        freeze_end = min(
                            freeze_start
                            + freeze_time,
                            source_part_end
                        )


                        freeze_duration_local = (
                            freeze_end
                            -
                            freeze_start
                        )


                        if (
                            freeze_duration_local
                            >
                            0.0001
                        ):

                            freeze_label = (
                                f"freeze_"
                                f"{part_index}_"
                                f"{piece_index}"
                            )


                            frame_time = max(
                                0.0,
                                freeze_start
                                - 0.0334
                            )


                            zoom_frames = max(
                                1,
                                int(
                                    math.ceil(
                                        freeze_duration_local
                                        * 30
                                    )
                                )
                            )


                            filter_parts.append(
                                f"[0:v]"
                                f"trim="
                                f"start={frame_time:.6f}:"
                                f"end={frame_time + 0.0334:.6f},"
                                f"setpts=PTS-STARTPTS,"
                                f"select='eq(n,0)',"
                                f"scale=576:1024,"
                                f"setsar=1,"
                                f"zoompan="
                                f"z='if(lte(on,29),"
                                f"1+0.15*on/29,"
                                f"1.15-0.15*(on-29)/29)':"
                                f"d={zoom_frames}:"
                                f"x='iw/2-(iw/zoom/2)':"
                                f"y='ih/2-(ih/zoom/2)':"
                                f"s=576x1024:"
                                f"fps=30,"
                                f"trim="
                                f"duration={freeze_duration_local:.6f},"
                                f"setpts=PTS-STARTPTS"
                                f"[{freeze_label}]"
                            )


                            video_labels.append(
                                f"[{freeze_label}]"
                            )


                            piece_index += 1


                        cursor = (
                            freeze_end
                        )


                        next_freeze += (
                            interval
                        )


                    if video_labels:

                        concat_inputs = (
                            "".join(
                                video_labels
                            )
                        )


                        filter_parts.append(
                            f"{concat_inputs}"
                            f"concat="
                            f"n={len(video_labels)}:"
                            f"v=1:"
                            f"a=0:"
                            f"unsafe=1,"
                            f"format=yuv420p"
                            f"[basevideo]"
                        )


                    else:

                        raise RuntimeError(
                            "No video segments were generated."
                        )


                # =================================================
                # NO SOURCE VIDEO
                # =================================================

                else:

                    last_frame_start = max(
                        0.0,
                        video_duration
                        - 0.05
                    )


                    filter_parts.append(
                        "[0:v]"
                        f"trim="
                        f"start={last_frame_start:.6f}:"
                        f"end={video_duration:.6f},"
                        "setpts=PTS-STARTPTS,"
                        "scale=576:1024,"
                        "setsar=1,"
                        "format=yuv420p"
                        "[basevideo]"
                    )


                # =================================================
                # SUBTITLE OVERLAY
                # =================================================

                filter_parts.append(
                    "[basevideo]"
                    f"ass=filename='{ass_filter_file}'"
                    f":fontsdir='{ass_fonts_dir}'"
                    "[vsub]"
                )


                # =================================================
                # VIDEO DURATION IN CURRENT PART
                # =================================================

                available_video_duration = max(
                    0.0,
                    min(
                        part_end,
                        video_duration
                    )
                    -
                    part_start
                )


                video_extra_duration = max(
                    0.0,
                    part_duration
                    -
                    available_video_duration
                )


                # =================================================
                # HOLD LAST FRAME WHEN VOICEOVER IS LONGER
                # =================================================

                if video_extra_duration > 0:

                    filter_parts.append(
                        "[vsub]"
                        f"tpad="
                        f"stop_mode=clone:"
                        f"stop_duration="
                        f"{video_extra_duration:.3f},"
                        "setpts=PTS-STARTPTS"
                        "[vout]"
                    )

                    final_video_label = (
                        "[vout]"
                    )

                else:

                    final_video_label = (
                        "[vsub]"
                    )


                # =================================================
                # VOICEOVER FOR THIS PART
                # =================================================

                voice_start = min(
                    part_start,
                    voice_duration
                )

                voice_end = min(
                    part_end,
                    voice_duration
                )


                voice_part_duration = max(
                    0.0,
                    voice_end
                    -
                    voice_start
                )


                # =================================================
                # AUDIO FILTER
                # =================================================

                if voice_part_duration > 0:

                    audio_filter = (
                        f"[1:a:0]"
                        f"atrim="
                        f"start={voice_start:.6f}:"
                        f"end={voice_end:.6f},"
                        f"asetpts=PTS-STARTPTS"
                    )


                    audio_pad_duration = max(
                        0.0,
                        part_duration
                        -
                        voice_part_duration
                    )


                    if audio_pad_duration > 0:

                        audio_filter += (
                            f",apad="
                            f"pad_dur="
                            f"{audio_pad_duration:.3f}"
                        )


                    audio_filter += (
                        "[aout]"
                    )


                else:

                    audio_filter = (
                        "anullsrc="
                        "channel_layout=stereo:"
                        "sample_rate=48000,"
                        f"atrim="
                        f"duration={part_duration:.3f}"
                        "[aout]"
                    )


                filter_parts.append(
                    audio_filter
                )


                # =================================================
                # FILTER COMPLEX
                # =================================================

                filter_complex = ";".join(
                    filter_parts
                )


                # =================================================
                # PART OUTPUT FILE
                # =================================================

                part_output = os.path.join(
                    export_dir,
                    f"rendered_part_{part_index:03d}.mp4"
                )


                # =================================================
                # FFMPEG COMMAND
                # =================================================

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
                    f"{part_duration:.3f}",

                    part_output
                ]


                # =================================================
                # RUN FFMPEG
                # =================================================

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True
                )


                if result.returncode != 0:

                    st.error(
                        f"❌ Part "
                        f"{part_index + 1}/{part_count} "
                        f"Export Failed"
                    )

                    st.code(
                        result.stderr[-6000:],
                        language="text"
                    )

                    raise RuntimeError(
                        f"Part "
                        f"{part_index + 1} export failed."
                    )


                if not os.path.isfile(
                    part_output
                ):

                    raise RuntimeError(
                        f"Part "
                        f"{part_index + 1} output file "
                        f"was not created."
                    )


                actual_part_duration = (
                    get_audio_duration(
                        part_output
                    )
                )


                part_files.append(
                    part_output
                )


                st.success(
                    f"✅ Part "
                    f"{part_index + 1}/{part_count} "
                    f"completed • "
                    f"{actual_part_duration:.2f} sec"
                )


            # =================================================
            # COMBINE PARTS
            # =================================================

            if len(part_files) == 1:

                output_video = (
                    part_files[0]
                )

                final_output = os.path.join(
                    tempfile.gettempdir(),
                    "final_movie_recap.mp4"
                )

                shutil.copyfile(
                    output_video,
                    final_output
                )

                output_video = (
                    final_output
                )


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
                            os.path.abspath(
                                part_file
                            )
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


                output_video = os.path.join(
                    tempfile.gettempdir(),
                    "final_movie_recap.mp4"
                )


                # =================================================
                # STREAM COPY MERGE
                # =================================================

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


                # =================================================
                # FALLBACK RE-ENCODE
                # =================================================

                if concat_result.returncode != 0:

                    st.warning(
                        "⚠️ Stream copy merge failed. "
                        "Trying compatible re-encode..."
                    )


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

                        output_video
                    ]


                    fallback_result = subprocess.run(
                        fallback_command,
                        capture_output=True,
                        text=True
                    )


                    if fallback_result.returncode != 0:

                        st.error(
                            "❌ Could not combine exported parts."
                        )

                        st.code(
                            fallback_result.stderr[-6000:],
                            language="text"
                        )

                        raise RuntimeError(
                            "Final video combination failed."
                        )


                st.success(
                    f"🔗 {len(part_files)} parts "
                    "combined successfully."
                )


            # =================================================
            # FINAL DURATION
            # =================================================

            final_duration = get_audio_duration(
                output_video
            )


            # =================================================
            # FINAL SYNC INFORMATION
            # =================================================

            duration_difference = abs(
                final_duration
                -
                voice_duration
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


            if duration_difference <= 0.50:

                st.success(
                    "✅ Video and Voiceover timeline "
                    "are synchronized."
                )

            else:

                st.warning(
                    f"⚠️ Final duration difference: "
                    f"{duration_difference:.2f} sec"
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
                "🧊 Freeze + Zoom is timeline-preserving. "
                "It does not add extra seconds to the final video."
            )


            st.info(
                "🤖 Gemini API: "
                "No additional Gemini calls during export."
            )


            st.info(
                "🔄 Video + Voiceover + Subtitle "
                "use the same final timeline."
            )


            # =================================================
            # READ FINAL VIDEO
            # =================================================

            with open(
                output_video,
                "rb"
            ) as f:

                final_bytes = f.read()


            # =================================================
            # PREVIEW
            # =================================================

            st.video(
                final_bytes
            )


            # =================================================
            # DOWNLOAD
            # =================================================

            st.download_button(
                "⬇️ Download Final Video",
                final_bytes,
                file_name=(
                    "final_movie_recap.mp4"
                ),
                mime="video/mp4"
            )


        except Exception as e:

            st.error(
                f"❌ Final Video Export Error: {e}"
            )
