import streamlit as st
import os
from google import genai
from google.genai import types
import cv2
import tempfile
import subprocess
import math
import asyncio
import re
import shutil


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Movie Recap AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "processing": False,
    "process_complete": False,
    "process_error": None,
    "process_percent": 0,
    "process_step": "",
    "final_video": None,

    "uploaded_file": None,
    "video_path": None,
    "video_duration": 0.0,

    "transcript_result": None,
    "transcript": None,

    "ai_scene_analysis": None,
    "scene_analysis_frames": 0,

    "recap_script": None,
    "myanmar_recap": None,

    "voiceover_file": None,
    "voice_duration": 0.0,
    "subtitle_data": None,
    "subtitle_timing_source": None,
    "voice_chunks": None,
    "voice_speed_value": 1.0,
    "selected_voice": None,
    "tts_rate": None,

    "final_video_duration": 0.0,

    # Chunk processing
    "total_parts": 1,
    "current_part": 0,
    "part_files": [],
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# COLOR THEME / CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(88, 28, 135, 0.28),
                transparent 35%
            ),
            radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.22),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #070b16 0%,
                #0b1020 45%,
                #111827 100%
            );

        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;

        background:
            linear-gradient(
                90deg,
                #a855f7,
                #6366f1,
                #38bdf8
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        text-shadow:
            0 0 30px rgba(99, 102, 241, 0.25);
    }

    h2 {
        color: #e9d5ff !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #c4b5fd !important;
        font-weight: 700 !important;
    }

    p,
    label,
    .stMarkdown,
    .stCaption {
        color: #e5e7eb;
    }

    hr {
        border: none !important;
        height: 1px !important;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(139, 92, 246, 0.8),
                rgba(59, 130, 246, 0.8),
                transparent
            ) !important;

        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }

    [data-testid="stFileUploader"] {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.90),
                rgba(15, 23, 42, 0.95)
            );

        border: 1px solid rgba(139, 92, 246, 0.45);
        border-radius: 18px;
        padding: 12px;

        box-shadow:
            0 8px 30px rgba(0, 0, 0, 0.35),
            0 0 25px rgba(99, 102, 241, 0.08);
    }

    [data-testid="stFileUploaderDropzone"] {
        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.75),
                rgba(49, 46, 129, 0.30)
            ) !important;

        border: 1px dashed rgba(167, 139, 250, 0.7) !important;
        border-radius: 14px !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;

        border: 1px solid rgba(139, 92, 246, 0.5);

        background:
            linear-gradient(
                135deg,
                #6d28d9,
                #4f46e5
            );

        color: white;
        font-weight: 700;
        padding: 0.65rem 1rem;

        box-shadow:
            0 6px 20px rgba(79, 70, 229, 0.25);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #c4b5fd;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #2563eb
            );

        transform: translateY(-1px);

        box-shadow:
            0 8px 25px rgba(99, 102, 241, 0.40);
    }

    button[kind="primary"] {
        background:
            linear-gradient(
                90deg,
                #7c3aed,
                #4f46e5,
                #2563eb
            ) !important;

        border: 1px solid rgba(196, 181, 253, 0.65) !important;

        font-size: 1rem !important;
        font-weight: 800 !important;

        box-shadow:
            0 8px 30px rgba(79, 70, 229, 0.40) !important;
    }

    button[kind="primary"]:hover {
        background:
            linear-gradient(
                90deg,
                #8b5cf6,
                #6366f1,
                #3b82f6
            ) !important;

        transform: translateY(-2px);
    }

    div[data-baseweb="select"] > div {
        background:
            rgba(15, 23, 42, 0.92) !important;

        border:
            1px solid rgba(139, 92, 246, 0.35) !important;

        border-radius:
            10px !important;
    }

    input,
    textarea {
        background:
            rgba(15, 23, 42, 0.92) !important;

        color:
            #f8fafc !important;

        border:
            1px solid rgba(139, 92, 246, 0.35) !important;

        border-radius:
            10px !important;
    }

    [data-testid="stTextArea"] textarea {
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.96),
                rgba(30, 41, 59, 0.90)
            ) !important;

        border:
            1px solid rgba(99, 102, 241, 0.35) !important;

        line-height:
            1.65 !important;
    }

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.92),
                rgba(49, 46, 129, 0.25)
            );

        border:
            1px solid rgba(139, 92, 246, 0.30);

        border-radius: 16px;
        padding: 15px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.25);
    }

    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
        border:
            1px solid rgba(139, 92, 246, 0.25);
    }

    video {
        border-radius: 16px !important;

        border:
            1px solid rgba(99, 102, 241, 0.35) !important;

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.45) !important;
    }

    audio {
        width: 100%;
        border-radius: 12px;
        margin-top: 8px;
    }

    div[data-testid="stCheckbox"] {
        background:
            rgba(30, 41, 59, 0.55);

        border-radius: 10px;
        padding: 8px 12px;
    }

    div[data-testid="stSlider"] {
        background:
            rgba(30, 41, 59, 0.45);

        padding: 10px 14px;
        border-radius: 12px;
    }

    .success-box {
        background:
            linear-gradient(
                135deg,
                rgba(16, 185, 129, 0.15),
                rgba(5, 150, 105, 0.08)
            );

        border:
            1px solid rgba(16, 185, 129, 0.35);

        border-radius: 14px;
        padding: 14px;
        margin: 10px 0;
    }

    .info-box {
        background:
            linear-gradient(
                135deg,
                rgba(59, 130, 246, 0.15),
                rgba(37, 99, 235, 0.08)
            );

        border:
            1px solid rgba(59, 130, 246, 0.35);

        border-radius: 14px;
        padding: 14px;
        margin: 10px 0;
    }

    .warning-box {
        background:
            linear-gradient(
                135deg,
                rgba(245, 158, 11, 0.15),
                rgba(217, 119, 6, 0.08)
            );

        border:
            1px solid rgba(245, 158, 11, 0.35);

        border-radius: 14px;
        padding: 14px;
        margin: 10px 0;
    }

    .processing-card {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.92),
                rgba(49, 46, 129, 0.30)
            );

        border:
            1px solid rgba(139, 92, 246, 0.40);

        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;

        box-shadow:
            0 15px 50px rgba(0, 0, 0, 0.35);
    }

    .complete-card {
        background:
            linear-gradient(
                145deg,
                rgba(16, 185, 129, 0.16),
                rgba(49, 46, 129, 0.20)
            );

        border:
            1px solid rgba(52, 211, 153, 0.40);

        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;

        box-shadow:
            0 15px 50px rgba(0, 0, 0, 0.35);
    }

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 1rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        h3 {
            font-size: 1.15rem !important;
        }

        .stButton > button {
            min-height: 48px;
        }

        [data-testid="stMetric"] {
            padding: 10px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_audio_duration(media_file):

    try:

        if not os.path.exists(media_file):
            return 0.0

        if os.path.getsize(media_file) <= 0:
            return 0.0

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

        duration = float(value)

        if duration <= 0:
            return 0.0

        return duration

    except Exception:
        return 0.0


def ass_time(seconds):

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
                    chunks.append(current)

                current = word

        if current:
            chunks.append(current)

    return chunks


def wrap_myanmar(
    text,
    max_chars=20
):

    text = str(text).strip()

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if len(text) <= max_chars:
        return text

    words = text.split()

    if len(words) > 1:

        best_split = None
        best_score = None

        for i in range(
            1,
            len(words)
        ):

            line1 = " ".join(
                words[:i]
            )

            line2 = " ".join(
                words[i:]
            )

            longest = max(
                len(line1),
                len(line2)
            )

            difference = abs(
                len(line1)
                - len(line2)
            )

            score = (
                longest * 2
                + difference
            )

            if (
                best_score is None
                or
                score < best_score
            ):

                best_score = score
                best_split = (
                    line1,
                    line2
                )

        if best_split:

            return (
                best_split[0]
                + "\\N"
                + best_split[1]
            )

    midpoint = math.ceil(
        len(text) / 2
    )

    punctuation_positions = [
        text.rfind(
            "၊",
            0,
            midpoint + 5
        ),
        text.rfind(
            " ",
            0,
            midpoint + 5
        )
    ]

    valid_positions = [
        p
        for p in punctuation_positions
        if p > 0
    ]

    if valid_positions:

        split_position = max(
            valid_positions
        )

    else:

        split_position = midpoint

    line1 = text[:split_position].strip()
    line2 = text[split_position:].strip()

    if line1 and line2:

        return (
            line1
            + "\\N"
            + line2
        )

    return text


def extract_frame_bytes(
    cap,
    timestamp
):

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

            new_w = int(w * scale)
            new_h = int(h * scale)

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


@st.cache_resource
def load_whisper_model(
    model_name="base"
):

    import whisper

    return whisper.load_model(
        model_name
    )


# =========================================================
# VIDEO SPLIT
# =========================================================

def split_video_into_parts(
    video_path,
    video_duration,
    work_root
):

    """
    Split video into exact 60-second parts.

    <= 60 sec:
        one part

    > 60 sec:
        60 sec chunks
    """

    os.makedirs(
        work_root,
        exist_ok=True
    )

    if video_duration <= 60.0:

        return [
            video_path
        ]

    total_parts = int(
        math.ceil(
            video_duration / 60.0
        )
    )

    part_files = []

    for index in range(total_parts):

        start_time = index * 60.0

        duration = min(
            60.0,
            video_duration - start_time
        )

        part_file = os.path.join(
            work_root,
            f"part_{index + 1:03d}.mp4"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            video_path,
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-avoid_negative_ts",
            "make_zero",
            part_file
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:

            raise RuntimeError(
                f"❌ Could not create Part {index + 1}:\n"
                f"{result.stderr[-4000:]}"
            )

        if (
            not os.path.exists(part_file)
            or
            os.path.getsize(part_file) < 1000
        ):

            raise RuntimeError(
                f"❌ Part {index + 1} was not created."
            )

        part_files.append(
            part_file
        )

    return part_files


# =========================================================
# PROCESSING UI
# =========================================================

def create_processing_ui():

    st.markdown(
        """
        <div class="processing-card">

            <div style="
                text-align:center;
            ">

                <div style="
                    font-size:4rem;
                ">
                    🎬
                </div>

                <h2>
                    Movie Recap AI
                </h2>

                <p>
                    Your movie recap is being created...
                </p>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    percent_text = st.empty()

    progress_bar = st.progress(0)

    current_step = st.empty()

    st.markdown("---")

    checklist = st.empty()

    return (
        percent_text,
        progress_bar,
        current_step,
        checklist
    )


def update_processing_ui(
    percent,
    step,
    percent_text,
    progress_bar,
    current_step,
    checklist
):

    percent = max(
        0,
        min(
            100,
            int(percent)
        )
    )

    st.session_state.process_percent = percent
    st.session_state.process_step = step

    progress_bar.progress(
        percent
    )

    percent_text.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:2rem;
            font-weight:800;
            color:#a78bfa;
            margin:15px 0;
        ">
            {percent}%
        </div>
        """,
        unsafe_allow_html=True
    )

    current_step.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:1.1rem;
            color:#e5e7eb;
            padding:10px;
        ">
            ⚙️ {step}
        </div>
        """,
        unsafe_allow_html=True
    )

    stages = [
        (5, "🎥 Video"),
        (20, "📝 Transcript"),
        (35, "🎬 Scene Analysis"),
        (48, "📝 Recap Script"),
        (60, "🇲🇲 Myanmar Translation"),
        (75, "🎙️ Myanmar Voiceover"),
        (85, "💬 Myanmar Subtitle"),
        (94, "🧊 Freeze + Zoom"),
        (100, "🎬 Final Export"),
    ]

    html = ""

    for stage_percent, name in stages:

        if percent >= stage_percent:

            icon = "✅"
            color = "#34d399"

        elif stage_percent <= percent + 10:

            icon = "🔄"
            color = "#a78bfa"

        else:

            icon = "○"
            color = "#64748b"

        html += f"""
        <div style="
            padding:10px 15px;
            margin:5px 0;
            border-radius:10px;
            background:rgba(30,41,59,0.55);
            color:{color};
            font-weight:600;
        ">
            {icon} {name}
        </div>
        """

    checklist.markdown(
        html,
        unsafe_allow_html=True
    )


# =========================================================
# COMPLETE SCREEN
# =========================================================

def show_complete_screen():

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:25px 10px;
        ">

            <div style="
                font-size:4rem;
            ">
                🎉
            </div>

            <h1>
                100% COMPLETE
            </h1>

            <p style="
                font-size:1.2rem;
                color:#c4b5fd;
            ">
                Your Movie Recap is Ready!
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="complete-card">

        <b>✅ Video uploaded</b><br><br>

        <b>✅ Transcript generated</b><br><br>

        <b>✅ Scene Analysis completed</b><br><br>

        <b>✅ Recap Script created</b><br><br>

        <b>✅ Myanmar translation completed</b><br><br>

        <b>✅ Myanmar Voiceover generated</b><br><br>

        <b>✅ Myanmar Subtitle synchronized</b><br><br>

        <b>✅ Freeze Frame + Zoom applied</b><br><br>

        <b>✅ Final Video exported</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    final_video = st.session_state.get(
        "final_video"
    )

    if (
        final_video
        and
        os.path.exists(final_video)
    ):

        st.subheader(
            "▶️ Preview Final Video"
        )

        st.video(
            final_video
        )

        with open(
            final_video,
            "rb"
        ) as f:

            video_data = f.read()

        st.download_button(
            "⬇️ DOWNLOAD FINAL VIDEO",
            data=video_data,
            file_name="final_movie_recap.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True
        )


# =========================================================
# PROCESS ONE VIDEO PART
# =========================================================

def process_one_part(
    chunk_path,
    part_index,
    total_parts,
    work_root,
    percent_text,
    progress_bar,
    current_step,
    checklist
):

    # =====================================================
    # SETTINGS
    # =====================================================

    whisper_model = st.session_state[
        "main_whisper_model"
    ]

    voice_gender = st.session_state[
        "main_voice_gender"
    ]

    voice_speed = st.session_state[
        "main_voice_speed"
    ]

    freeze_enabled = st.session_state[
        "main_freeze_enabled"
    ]

    freeze_interval = st.session_state[
        "main_freeze_interval"
    ]

    freeze_duration = st.session_state[
        "main_freeze_duration"
    ]


    part_duration = get_audio_duration(
        chunk_path
    )

    if part_duration <= 0:

        raise RuntimeError(
            f"Part {part_index} has invalid duration."
        )


    # =====================================================
    # PART PROGRESS RANGE
    # =====================================================

    part_start_percent = (
        (part_index - 1)
        /
        total_parts
        *
        90
    )

    part_end_percent = (
        part_index
        /
        total_parts
        *
        90
    )


    def part_progress(
        local_percent,
        message
    ):

        local_percent = max(
            0,
            min(
                100,
                local_percent
            )
        )

        overall = (
            part_start_percent
            +
            (
                local_percent
                /
                100
            )
            *
            (
                part_end_percent
                -
                part_start_percent
            )
        )

        update_processing_ui(
            int(overall),
            f"Part {part_index}/{total_parts} — {message}",
            percent_text,
            progress_bar,
            current_step,
            checklist
        )


    # =====================================================
    # 1. TRANSCRIPT
    # =====================================================

    part_progress(
        5,
        "🎥 Preparing video..."
    )

    part_progress(
        15,
        f"📝 Transcribing with Whisper {whisper_model}..."
    )

    model = load_whisper_model(
        whisper_model
    )

    result = model.transcribe(
        chunk_path,
        language="en"
    )

    transcript = result.get(
        "text",
        ""
    )

    if not transcript.strip():

        raise RuntimeError(
            f"Part {part_index}: Whisper returned empty transcript."
        )


    # =====================================================
    # 2. SCENE ANALYSIS
    # =====================================================

    part_progress(
        25,
        "🎬 Analyzing actual video scenes..."
    )

    segments = result.get(
        "segments",
        []
    )

    if not segments:

        raise RuntimeError(
            f"Part {part_index}: No timestamped transcript segments found."
        )


    api_key = st.secrets[
        "GEMINI_API_KEY"
    ]

    client = genai.Client(
        api_key=api_key
    )


    cap = cv2.VideoCapture(
        chunk_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Part {part_index}: Could not open video."
        )


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


    if len(usable_segments) > max_frames:

        selected_indexes = []

        for n in range(max_frames):

            pos = (
                n
                *
                (
                    len(usable_segments) - 1
                )
                /
                (
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

        selected_segments = usable_segments


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
            start
            +
            end
        ) / 2


        frame_bytes = extract_frame_bytes(
            cap,
            timestamp
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

        raise RuntimeError(
            f"Part {part_index}: No video frames could be extracted."
        )


    # =====================================================
    # ORIGINAL GEMINI SCENE PROMPT
    # =====================================================

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


    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents
    )


    scene_analysis = (
        response.text
        if response
        else ""
    )


    if not scene_analysis:

        raise RuntimeError(
            f"Part {part_index}: Gemini returned no scene analysis."
        )


    # =====================================================
    # 3. RECAP SCRIPT
    # =====================================================

    part_progress(
        40,
        "📝 Writing movie recap script..."
    )


    recap_prompt = f"""
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


    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=recap_prompt
    )


    recap_script = (
        response.text
        if response
        else ""
    )


    if not recap_script:

        raise RuntimeError(
            f"Part {part_index}: Gemini returned no recap script."
        )


    # =====================================================
    # 4. MYANMAR TRANSLATION
    # =====================================================

    part_progress(
        55,
        "🇲🇲 Translating recap into Myanmar..."
    )


    myanmar_prompt = f"""
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

English Recap:

{recap_script}
"""


    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=mymanmar_prompt
    )


    myanmar_text = (
        response.text
        if response
        else ""
    )


    if not myanmar_text:

        raise RuntimeError(
            f"Part {part_index}: Gemini returned no Myanmar narration."
        )


    # =====================================================
    # 5. MYANMAR VOICEOVER
    # =====================================================

    part_progress(
        65,
        "🎙️ Generating Myanmar voiceover..."
    )


    import edge_tts


    text = myanmar_text.strip()


    if not text:

        raise RuntimeError(
            f"Part {part_index}: Myanmar recap is empty."
        )


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


    chunks = split_myanmar_text(
        text,
        max_chars=100
    )


    if not chunks:

        raise RuntimeError(
            f"Part {part_index}: Could not split Myanmar narration."
        )


    voice_work_dir = os.path.join(
        work_root,
        f"voice_part_{part_index:03d}"
    )


    os.makedirs(
        voice_work_dir,
        exist_ok=True
    )


    async def create_all_tts():

        results = []

        for index, chunk in enumerate(chunks):

            raw_file = os.path.join(
                voice_work_dir,
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


            if (
                not os.path.exists(raw_file)
                or
                os.path.getsize(raw_file) < 1000
            ):

                raise RuntimeError(
                    f"Part {part_index}: "
                    f"TTS segment {index + 1} failed."
                )


            results.append(
                raw_file
            )


        return results


    raw_files = asyncio.run(
        create_all_tts()
    )


    raw_durations = []


    for index, raw_file in enumerate(raw_files):

        duration = get_audio_duration(
            raw_file
        )


        if duration <= 0:

            raise RuntimeError(
                f"Part {part_index}: "
                f"TTS segment {index + 1} has 0 duration."
            )


        raw_durations.append(
            duration
        )


    TTS_CROSSFADE = 0.08


    normalized_files = []


    for index, raw_file in enumerate(raw_files):

        normalized_file = os.path.join(
            voice_work_dir,
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
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                normalized_file
            ],
            capture_output=True,
            text=True
        )


        if normalize_result.returncode != 0:

            raise RuntimeError(
                "TTS normalization failed:\n"
                f"{normalize_result.stderr[-3000:]}"
            )


        normalized_files.append(
            normalized_file
        )


    combined_audio = os.path.join(
        voice_work_dir,
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

            output_label = f"[a{i}]"


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


            previous_label = output_label


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
                "TTS audio combining failed:\n"
                f"{combine_result.stderr[-3000:]}"
            )


    combined_duration = get_audio_duration(
        combined_audio
    )


    if combined_duration <= 0:

        raise RuntimeError(
            f"Part {part_index}: Combined TTS has 0 duration."
        )


    final_voice_file = os.path.join(
        voice_work_dir,
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
            "Final MP3 conversion failed:\n"
            f"{convert_result.stderr[-3000:]}"
        )


    voice_duration = get_audio_duration(
        final_voice_file
    )


    if voice_duration <= 0:

        raise RuntimeError(
            f"Part {part_index}: Final voiceover has 0 duration."
        )


    # =====================================================
    # SUBTITLE TIMING
    # =====================================================

    subtitle_data = []

    current_time = 0.0


    for index, chunk in enumerate(chunks):

        raw_duration = raw_durations[index]

        start_time = current_time

        end_time = (
            start_time
            +
            raw_duration
        )


        if index < len(chunks) - 1:

            next_time = (
                end_time
                -
                TTS_CROSSFADE
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
            -
            TTS_CROSSFADE
        )


        current_time = max(
            current_time,
            0
        )


    # =====================================================
    # FIX SUBTITLES
    # =====================================================

    part_progress(
        85,
        "💬 Synchronizing Myanmar subtitles..."
    )


    fixed_subtitles = []


    for item in subtitle_data:

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


        if start >= voice_duration:
            continue


        end = min(
            end,
            voice_duration
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


    # =====================================================
    # ASS SUBTITLE
    # =====================================================

    ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 576
PlayResY: 1024
ScaledBorderAndShadow: yes

[V4+Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Myanmar,Noto Sans Myanmar,68,&H0000FFFF,&H0000FFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,22,22,55,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


    for item in fixed_subtitles:

        new_start = max(
            0,
            float(item["start"])
        )


        new_end = max(
            new_start + 0.1,
            float(item["end"])
        )


        text = wrap_myanmar(
            item["text"],
            16
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
        work_root,
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


    # =====================================================
    # FREEZE + ZOOM
    # =====================================================

    part_progress(
        94,
        "🧊 Applying Freeze Frame + Zoom..."
    )


    if freeze_enabled:

        interval = float(
            freeze_interval
        )

        freeze_time = float(
            freeze_duration
        )

    else:

        interval = (
            part_duration + 1
        )

        freeze_time = 0


    filter_parts = []

    labels = []


    if freeze_enabled:

        segment_count = int(
            math.ceil(
                part_duration
                /
                interval
            )
        )


        for i in range(
            segment_count
        ):

            start = (
                i
                *
                interval
            )


            end = min(
                (i + 1)
                *
                interval,
                part_duration
            )


            normal_label = (
                f"normal{i}"
            )


            normal_filter = (
                f"[0:v]"
                f"trim="
                f"start={start}:"
                f"end={end},"
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


            if end < part_duration:

                freeze_label = (
                    f"freeze{i}"
                )


                frame_time = max(
                    start,
                    end - 0.10
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


    # =====================================================
    # ASS SUBTITLE BURN
    # =====================================================

    filter_parts.append(
        "[basevideo]"
        f"ass={ass_file}"
        "[vout]"
    )


    if freeze_enabled:

        actual_freeze_count = max(
            0,
            segment_count - 1
        )

    else:

        actual_freeze_count = 0


    base_video_duration = (
        part_duration
        +
        (
            actual_freeze_count
            *
            freeze_time
        )
    )


    extra_duration = max(
        0.0,
        voice_duration
        -
        base_video_duration
    )


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


    filter_complex = ";".join(
        filter_parts
    )


    # =====================================================
    # EXPORT THIS PART
    # =====================================================

    part_progress(
        97,
        "🎬 Exporting this part..."
    )


    output_video = os.path.join(
        work_root,
        f"exported_part_{part_index:03d}.mp4"
    )


    command = [
        "ffmpeg",
        "-y",
        "-i",
        chunk_path,
        "-i",
        final_voice_file,
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
        f"{max(voice_duration, base_video_duration):.3f}",
        output_video
    ]


    export_result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )


    if export_result.returncode != 0:

        raise RuntimeError(
            f"Part {part_index} FFmpeg Export Failed:\n"
            f"{export_result.stderr[-5000:]}"
        )


    if (
        not os.path.exists(output_video)
        or
        os.path.getsize(output_video) < 1000
    ):

        raise RuntimeError(
            f"Part {part_index} final video was not created."
        )


    return output_video


# =========================================================
# COMBINE EXPORTED PARTS
# =========================================================

def combine_exported_parts(
    part_files,
    work_root,
    percent_text,
    progress_bar,
    current_step,
    checklist
):

    update_processing_ui(
        96,
        "🔗 Combining all exported parts...",
        percent_text,
        progress_bar,
        current_step,
        checklist
    )


    if not part_files:

        raise RuntimeError(
            "No exported parts were found."
        )


    if len(part_files) == 1:

        final_video = os.path.join(
            work_root,
            "final_movie_recap.mp4"
        )

        shutil.copyfile(
            part_files[0],
            final_video
        )

        return final_video


    concat_file = os.path.join(
        work_root,
        "concat_parts.txt"
    )


    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:

        for part in part_files:

            absolute_path = os.path.abspath(
                part
            ).replace(
                "\\",
                "/"
            )

            absolute_path = absolute_path.replace(
                "'",
                "'\\''"
            )

            f.write(
                f"file '{absolute_path}'\n"
            )


    final_video = os.path.join(
        work_root,
        "final_movie_recap.mp4"
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
        final_video
    ]


    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )


    if result.returncode != 0:

        # Fallback: re-encode if stream copy cannot concatenate
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
            final_video
        ]


        fallback_result = subprocess.run(
            fallback_command,
            capture_output=True,
            text=True
        )


        if fallback_result.returncode != 0:

            raise RuntimeError(
                "❌ Could not combine exported parts:\n"
                f"{fallback_result.stderr[-5000:]}"
            )


    if (
        not os.path.exists(final_video)
        or
        os.path.getsize(final_video) < 1000
    ):

        raise RuntimeError(
            "Final combined video was not created."
        )


    return final_video


# =========================================================
# MAIN PROCESSING PIPELINE
# =========================================================

def process_movie_recap(
    percent_text,
    progress_bar,
    current_step,
    checklist
):

    video_path = st.session_state[
        "video_path"
    ]


    if not video_path:

        raise RuntimeError(
            "Video path is missing."
        )


    if not os.path.exists(video_path):

        raise RuntimeError(
            "Uploaded video file could not be found."
        )


    # =====================================================
    # ORIGINAL VIDEO DURATION
    # =====================================================

    video_duration = get_audio_duration(
        video_path
    )


    if video_duration <= 0:

        raise RuntimeError(
            "Could not determine video duration."
        )


    st.session_state[
        "video_duration"
    ] = video_duration


    # =====================================================
    # CREATE WORK DIRECTORY
    # =====================================================

    work_root = tempfile.mkdtemp(
        prefix="movie_recap_job_"
    )


    # =====================================================
    # SPLIT VIDEO
    # =====================================================

    update_processing_ui(
        3,
        "🎥 Checking video duration...",
        percent_text,
        progress_bar,
        current_step,
        checklist
    )


    if video_duration > 60.0:

        total_parts = int(
            math.ceil(
                video_duration / 60.0
            )
        )


        update_processing_ui(
            5,
            (
                f"✂️ Video is {video_duration:.1f}s. "
                f"Splitting into {total_parts} parts..."
            ),
            percent_text,
            progress_bar,
            current_step,
            checklist
        )


        part_files = split_video_into_parts(
            video_path,
            video_duration,
            work_root
        )

    else:

        total_parts = 1

        part_files = [
            video_path
        ]


    st.session_state[
        "total_parts"
    ] = total_parts


    st.session_state[
        "current_part"
    ] = 0


    # =====================================================
    # PROCESS EACH PART
    # =====================================================

    exported_parts = []


    for index, part_file in enumerate(
        part_files,
        start=1
    ):

        st.session_state[
            "current_part"
        ] = index


        update_processing_ui(
            int(
                (
                    (index - 1)
                    /
                    total_parts
                )
                *
                90
            ),
            (
                f"🎬 Processing Part "
                f"{index}/{total_parts}"
            ),
            percent_text,
            progress_bar,
            current_step,
            checklist
        )


        exported_part = process_one_part(
            part_file,
            index,
            total_parts,
            work_root,
            percent_text,
            progress_bar,
            current_step,
            checklist
        )


        exported_parts.append(
            exported_part
        )


        st.session_state[
            "part_files"
        ] = exported_parts.copy()


    # =====================================================
    # COMBINE ALL PARTS
    # =====================================================

    update_processing_ui(
        95,
        "🔗 All parts exported. Combining...",
        percent_text,
        progress_bar,
        current_step,
        checklist
    )


    final_video = combine_exported_parts(
        exported_parts,
        work_root,
        percent_text,
        progress_bar,
        current_step,
        checklist
    )


    # =====================================================
    # FINAL DURATION
    # =====================================================

    final_duration = get_audio_duration(
        final_video
    )


    if final_duration <= 0:

        raise RuntimeError(
            "Final video duration is invalid."
        )


    st.session_state[
        "final_video_duration"
    ] = final_duration


    st.session_state[
        "final_video"
    ] = final_video


    # =====================================================
    # COMPLETE
    # =====================================================

    update_processing_ui(
        100,
        "🎉 Movie Recap completed successfully!",
        percent_text,
        progress_bar,
        current_step,
        checklist
    )


    return final_video


# =========================================================
# HEADER
# =========================================================

st.title(
    "🎬 Movie Recap AI"
)

st.write(
    "Upload a movie and generate a complete Myanmar movie recap."
)


# =========================================================
# COMPLETE SCREEN FIRST
# =========================================================

if (
    st.session_state.process_complete
    and
    st.session_state.final_video
):

    show_complete_screen()

    st.stop()


# =========================================================
# PROCESSING SCREEN
# =========================================================

if st.session_state.processing:

    (
        percent_text,
        progress_bar,
        current_step,
        checklist
    ) = create_processing_ui()


    try:

        output_video = process_movie_recap(
            percent_text,
            progress_bar,
            current_step,
            checklist
        )


        st.session_state.final_video = (
            output_video
        )


        st.session_state.processing = False

        st.session_state.process_complete = True

        st.session_state.process_error = None

        st.session_state.process_percent = 100


        st.rerun()


    except Exception as e:

        st.session_state.processing = False

        st.session_state.process_complete = False

        st.session_state.process_error = str(e)


        st.error(
            f"❌ Movie Recap Failed\n\n{e}"
        )


        st.stop()


# =========================================================
# NORMAL SETUP UI
# =========================================================

st.markdown(
    """
    <div class="info-box">

        🎥 <b>Video → Transcript → Scene Analysis
        → Recap Script → Myanmar Voiceover
        → Myanmar Subtitle → Freeze + Zoom
        → Final Video</b>

        <br><br>

        ⏱️ <b>Videos longer than 1 minute are
        automatically split into 60-second parts.</b>

    </div>
    """,
    unsafe_allow_html=True
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
    ],
    key="movie_upload"
)


if uploaded_file is not None:

    video_bytes = uploaded_file.getvalue()


    file_size_mb = (
        uploaded_file.size
        /
        (1024 * 1024)
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
        "uploaded_file"
    ] = video_bytes


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
                frame_count
                /
                fps
            )

        else:

            duration_seconds = get_audio_duration(
                video_path
            )


        minutes = int(
            duration_seconds // 60
        )


        seconds = int(
            duration_seconds % 60
        )


        # Automatic part information

        if duration_seconds > 60:

            total_parts_preview = int(
                math.ceil(
                    duration_seconds / 60
                )
            )

        else:

            total_parts_preview = 1


        st.markdown(
            """
            <div class="success-box">
                ✅ <b>Video uploaded successfully!</b>
            </div>
            """,
            unsafe_allow_html=True
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


        if duration_seconds > 60:

            st.markdown(
                f"""
                <div class="info-box">

                ✂️ <b>Long Video Mode</b><br><br>

                This video is
                <b>{duration_seconds:.1f} seconds</b> long.

                <br><br>

                Movie Recap AI will automatically create
                <b>{total_parts_preview} parts</b>.

                <br><br>

                Each part will be processed separately
                for better Voiceover + Subtitle synchronization.

                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                """
                <div class="success-box">

                ⚡ Video is 1 minute or shorter.

                <br><br>

                Normal single-video processing will be used.

                </div>
                """,
                unsafe_allow_html=True
            )


        st.session_state[
            "video_duration"
        ] = duration_seconds


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
# RECAP SETTINGS
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


st.caption(
    "⚙️ Set your preferred settings first, "
    "then press One Click."
)


st.divider()


# =========================================================
# ONE CLICK
# =========================================================

if uploaded_file is not None:

    st.markdown(
        "### 🎬 Ready to Generate"
    )


    st.markdown(
        """
        <div class="info-box">

        Once you press <b>ONE CLICK</b>, Movie Recap AI
        will automatically run:

        <br><br>

        ✂️ Auto Split Long Video<br>
        🎤 Whisper Transcript<br>
        🎬 Gemini Scene Analysis<br>
        📝 Recap Script<br>
        🇲🇲 Myanmar Translation<br>
        🎙️ Myanmar Voiceover<br>
        💬 Subtitle Timing<br>
        🧊 Freeze Frame + Zoom<br>
        🎬 Part-by-Part Export<br>
        🔗 Final Part Combining

        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "🎬 ONE CLICK — GENERATE MOVIE RECAP",
        type="primary",
        use_container_width=True
    ):

        st.session_state.processing = True

        st.session_state.process_complete = False

        st.session_state.process_error = None

        st.session_state.process_percent = 0

        st.session_state.process_step = "Starting..."

        st.session_state.final_video = None

        st.session_state.part_files = []

        st.session_state.current_part = 0

        st.rerun()


else:

    st.info(
        "🎥 Please upload a movie/video first."
    )


# =========================================================
# ERROR DISPLAY
# =========================================================

if (
    st.session_state.process_error
    and
    not st.session_state.processing
):

    st.markdown(
        """
        <div class="warning-box">

            ⚠️ The previous processing attempt failed.
            Please check the error above and try again.

        </div>
        """,
        unsafe_allow_html=True
    )
