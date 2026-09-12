import streamlit as st
import os
import cv2
import tempfile
import subprocess
import re
import asyncio
import shutil

from google import genai
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Movie Recap AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recap AI")
st.write("Upload a movie and analyze video information.")


# =========================================================
# HELPERS
# =========================================================

def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-5000:]
        )

    return result


def get_media_duration(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path
            ],
            capture_output=True,
            text=True
        )

        return float(
            result.stdout.strip()
        )

    except Exception:
        return 0.0


def ass_time(seconds):
    seconds = max(
        0,
        float(seconds)
    )

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    cs = int(
        round(
            (seconds - int(seconds)) * 100
        )
    )

    if cs >= 100:
        s += 1
        cs = 0

    return (
        f"{h}:"
        f"{m:02d}:"
        f"{s:02d}."
        f"{cs:02d}"
    )


def split_myanmar_text(
    text,
    max_chars=65
):
    text = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()

    if not text:
        return []

    parts = re.split(
        r"(?<=[။!?])\s+|(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if (
            len(current)
            + len(part)
            + 1
            <= max_chars
        ):
            current = (
                current + " " + part
            ).strip()

        else:

            if current:
                chunks.append(
                    current
                )

            if len(part) <= max_chars:
                current = part

            else:

                for i in range(
                    0,
                    len(part),
                    max_chars
                ):

                    piece = part[
                        i:i + max_chars
                    ]

                    if (
                        i + max_chars
                        < len(part)
                    ):
                        chunks.append(
                            piece
                        )

                    else:
                        current = piece

    if current:
        chunks.append(current)

    return chunks


def wrap_myanmar(
    text,
    max_chars=24
):
    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    words = text.split()

    if len(words) == 1:

        lines = []
        current = ""

        for ch in text:

            if len(current) >= max_chars:
                lines.append(
                    current
                )
                current = ""

            current += ch

        if current:
            lines.append(
                current
            )

        return "\\N".join(
            lines[:3]
        )

    lines = []
    current = ""

    for word in words:

        if (
            len(current)
            + len(word)
            + 1
            <= max_chars
        ):
            current = (
                current + " " + word
            ).strip()

        else:

            if current:
                lines.append(
                    current
                )

            current = word

    if current:
        lines.append(current)

    return "\\N".join(
        lines[:3]
    )


def extract_preview_frame(
    video_path,
    time_seconds=0,
    width=576,
    height=1024
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return None

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        time_seconds * 1000
    )

    ok, frame = cap.read()

    if not ok:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        ok, frame = cap.read()

    cap.release()

    if not ok:
        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    h, w = frame.shape[:2]

    scale = min(
        width / w,
        height / h
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    frame = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "black"
    )

    x = (
        width - new_w
    ) // 2

    y = (
        height - new_h
    ) // 2

    canvas.paste(
        Image.fromarray(frame),
        (x, y)
    )

    return canvas


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
                "type": "rect",
                "version": "7.0.0",
                "left": float(x),
                "top": float(y),
                "width": float(width),
                "height": float(height),
                "fill": "rgba(255,0,0,0.25)",
                "stroke": "#FF0000",
                "strokeWidth": 4,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "originX": "left",
                "originY": "top",
                "skewX": 0,
                "skewY": 0,
                "selectable": True,
                "evented": True,
                "hasControls": True,
                "hasBorders": True,
                "lockRotation": True
            }
        ]
    }


def normalize_blur_box(
    x,
    y,
    width,
    height,
    canvas_w=576,
    canvas_h=1024
):

    x = max(
        0,
        min(
            float(x),
            canvas_w - 1
        )
    )

    y = max(
        0,
        min(
            float(y),
            canvas_h - 1
        )
    )

    width = max(
        10,
        min(
            float(width),
            canvas_w - x
        )
    )

    height = max(
        10,
        min(
            float(height),
            canvas_h - y
        )
    )

    return (
        int(round(x)),
        int(round(y)),
        int(round(width)),
        int(round(height))
    )


# =========================================================
# SUBTITLE
# =========================================================

def create_srt(subtitles):

    lines = []

    def srt_time(sec):

        sec = max(
            0,
            float(sec)
        )

        h = int(sec // 3600)
        m = int(
            (sec % 3600) // 60
        )
        s = int(sec % 60)

        ms = int(
            round(
                (
                    sec
                    - int(sec)
                ) * 1000
            )
        )

        if ms >= 1000:
            s += 1
            ms = 0

        return (
            f"{h:02d}:"
            f"{m:02d}:"
            f"{s:02d},"
            f"{ms:03d}"
        )

    for i, item in enumerate(
        subtitles,
        1
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

        lines.append(
            f"{i}\n"
            f"{srt_time(start)} --> "
            f"{srt_time(end)}\n"
            f"{text}\n"
        )

    return "\n".join(lines)


def create_ass(subtitles):

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 576
PlayResY: 1024
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Myanmar,Noto Sans Myanmar,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,25,25,55,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]

    for item in subtitles:

        start = ass_time(
            item["start"]
        )

        end = ass_time(
            item["end"]
        )

        text = wrap_myanmar(
            item["text"]
        )

        text = (
            text
            .replace(
                "{",
                "\\{"
            )
            .replace(
                "}",
                "\\}"
            )
        )

        lines.append(
            "Dialogue: "
            f"0,{start},{end},"
            f"Myanmar,,0,0,0,,"
            f"{text}"
        )

    return "\n".join(lines)


# =========================================================
# WHISPER
# =========================================================

@st.cache_resource
def load_whisper_model(
    model_name
):

    import whisper

    return whisper.load_model(
        model_name
    )


# =========================================================
# UPLOAD
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


if uploaded_file:

    file_size_mb = (
        uploaded_file.size
        / (1024 * 1024)
    )

    if file_size_mb > 200:

        st.error(
            "❌ Maximum file size is 200MB."
        )

        st.stop()

    temp_dir = tempfile.mkdtemp(
        prefix="movie_recap_"
    )

    input_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(
        input_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    st.success(
        f"✅ Uploaded: "
        f"{uploaded_file.name}"
    )

    cap = cv2.VideoCapture(
        input_path
    )

    if cap.isOpened():

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        duration = (
            frame_count / fps
            if fps
            else 0
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

        cap.release()

        st.info(
            f"🎬 Video Duration: "
            f"{duration:.1f} seconds"
        )

        st.write(
            f"📐 Resolution: "
            f"{width} × {height}"
        )

        st.write(
            f"🎞️ FPS: {fps:.2f}"
        )

        st.video(
            input_path
        )


        # =================================================
        # TRANSCRIPT
        # =================================================

        st.divider()

        st.subheader(
            "📝 Transcript"
        )

        whisper_size = st.selectbox(
            "Whisper Model",
            [
                "tiny",
                "base"
            ],
            index=0
        )

        if st.button(
            "🎙️ Generate Transcript",
            use_container_width=True
        ):

            with st.spinner(
                "🎙️ Transcribing..."
            ):

                try:

                    model = (
                        load_whisper_model(
                            whisper_size
                        )
                    )

                    result = model.transcribe(
                        input_path,
                        language="en",
                        fp16=False
                    )

                    transcript = (
                        result.get(
                            "text",
                            ""
                        )
                    )

                    st.session_state[
                        "transcript"
                    ] = transcript

                    st.success(
                        "✅ Transcript generated!"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Whisper Error: {e}"
                    )


        transcript = (
            st.session_state.get(
                "transcript",
                ""
            )
        )


        if transcript:

            st.text_area(
                "Transcript",
                transcript,
                height=200
            )


            # =================================================
            # SCENE ANALYSIS
            # =================================================

            st.divider()

            st.subheader(
                "🎬 Scene Analysis"
            )

            if st.button(
                "🔍 Analyze Scenes",
                use_container_width=True
            ):

                scenes = []

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    transcript
                )

                for i in range(
                    0,
                    len(sentences),
                    5
                ):

                    block = " ".join(
                        sentences[
                            i:i + 5
                        ]
                    ).strip()

                    if block:

                        scenes.append(
                            {
                                "scene":
                                len(scenes) + 1,

                                "text":
                                block
                            }
                        )

                st.session_state[
                    "scenes"
                ] = scenes

                st.success(
                    f"✅ {len(scenes)} "
                    f"scenes created."
                )


            scenes = (
                st.session_state.get(
                    "scenes",
                    []
                )
            )

            for scene in scenes:

                st.write(
                    f"### Scene "
                    f"{scene['scene']}"
                )

                st.write(
                    scene["text"]
                )


            # =================================================
            # AI SCENE ANALYSIS
            # =================================================

            st.divider()

            st.subheader(
                "🤖 AI Scene Analysis"
            )

            if st.button(
                "✨ Generate AI Scene Analysis",
                use_container_width=True
            ):

                try:

                    api_key = (
                        st.secrets[
                            "GEMINI_API_KEY"
                        ]
                    )

                    client = genai.Client(
                        api_key=api_key
                    )

                    prompt = f"""
Analyze the following movie transcript.

Create a chronological scene analysis.

Rules:
1. Do not invent events.
2. Use only information from the transcript.
3. Identify important actions.
4. Identify characters when possible.
5. Explain important visual moments.
6. Keep scenes suitable for recap video creation.

Transcript:

{transcript}
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
                        "✅ AI scene analysis generated!"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Gemini Error: {e}"
                    )


            ai_scene_analysis = (
                st.session_state.get(
                    "ai_scene_analysis",
                    ""
                )
            )

            if ai_scene_analysis:

                st.text_area(
                    "AI Scene Analysis",
                    ai_scene_analysis,
                    height=300
                )


            # =================================================
            # RECAP SCRIPT
            # =================================================

            st.divider()

            st.subheader(
                "📖 Recap Script"
            )

            if st.button(
                "📝 Generate Recap Script",
                use_container_width=True
            ):

                try:

                    api_key = (
                        st.secrets[
                            "GEMINI_API_KEY"
                        ]
                    )

                    client = genai.Client(
                        api_key=api_key
                    )

                    source = (
                        ai_scene_analysis
                        if ai_scene_analysis
                        else transcript
                    )

                    prompt = f"""
Create a movie recap narration script.

Rules:
1. Use only the provided information.
2. Never invent events.
3. Keep chronological order.
4. Make it engaging.
5. Make it suitable for voiceover.
6. Divide it into numbered scenes.
7. Each scene should contain narration.

Source:

{source}
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
                        "✅ Recap script generated!"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Gemini Error: {e}"
                    )


            recap_script = (
                st.session_state.get(
                    "recap_script",
                    ""
                )
            )

            if recap_script:

                st.text_area(
                    "Recap Script",
                    recap_script,
                    height=350
                )


                # =================================================
                # MYANMAR RECAP
                # =================================================

                st.divider()

                st.subheader(
                    "🇲🇲 Myanmar Recap"
                )

                if st.button(
                    "🇲🇲 Generate Myanmar Recap",
                    use_container_width=True
                ):

                    try:

                        api_key = (
                            st.secrets[
                                "GEMINI_API_KEY"
                            ]
                        )

                        client = genai.Client(
                            api_key=api_key
                        )

                        prompt = f"""
Translate and adapt the following movie
recap narration into natural Myanmar Burmese.

Rules:
1. Preserve exact meaning.
2. Do not invent information.
3. Keep chronological order.
4. Make it natural for Myanmar voiceover.
5. Keep numbered scenes.
6. Use Burmese ending "ဒယ်"
   instead of "တယ်" where natural.
7. Avoid overly formal wording.

Script:

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
                            "✅ Myanmar recap generated!"
                        )

                    except Exception as e:

                        st.error(
                            f"❌ Gemini Error: {e}"
                        )


                myanmar_recap = (
                    st.session_state.get(
                        "myanmar_recap",
                        ""
                    )
                )

                if myanmar_recap:

                    st.text_area(
                        "Myanmar Recap",
                        myanmar_recap,
                        height=350
                    )


                    # =================================================
                    # VOICEOVER
                    # =================================================

                    st.divider()

                    st.subheader(
                        "🎙️ Myanmar Female Voiceover"
                    )

                    voice_speed = st.select_slider(
                        "Voice Speed",
                        options=[
                            1.0,
                            1.1,
                            1.2
                        ],
                        value=1.0
                    )


                    if st.button(
                        "🎙️ Generate Myanmar Female Voiceover",
                        use_container_width=True
                    ):

                        async def generate_tts():

                            import edge_tts

                            text = re.sub(
                                r"Scene\s*\d+\s*:?",
                                "",
                                myanmar_recap,
                                flags=re.IGNORECASE
                            )

                            chunks = (
                                split_myanmar_text(
                                    text,
                                    65
                                )
                            )

                            mp3_files = []
                            durations = []

                            for i, chunk in enumerate(
                                chunks
                            ):

                                mp3_path = os.path.join(
                                    temp_dir,
                                    f"tts_{i:04d}.mp3"
                                )

                                communicate = (
                                    edge_tts.Communicate(
                                        chunk,
                                        "my-MM-NilarNeural"
                                    )
                                )

                                await communicate.save(
                                    mp3_path
                                )

                                mp3_files.append(
                                    mp3_path
                                )

                                durations.append(
                                    get_media_duration(
                                        mp3_path
                                    )
                                )


                            concat_file = os.path.join(
                                temp_dir,
                                "tts_concat.txt"
                            )

                            with open(
                                concat_file,
                                "w",
                                encoding="utf-8"
                            ) as f:

                                for path in mp3_files:

                                    f.write(
                                        f"file '{path}'\n"
                                    )


                            raw_voice = os.path.join(
                                temp_dir,
                                "voice_raw.mp3"
                            )

                            run_cmd(
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
                                    raw_voice
                                ]
                            )


                            final_voice = os.path.join(
                                temp_dir,
                                "myanmar_voice.mp3"
                            )

                            speed = float(
                                voice_speed
                            )

                            if speed == 1.0:

                                shutil.copy(
                                    raw_voice,
                                    final_voice
                                )

                            else:

                                filters = []

                                remaining = speed

                                while remaining > 2.0:

                                    filters.append(
                                        "atempo=2.0"
                                    )

                                    remaining /= 2.0

                                while remaining < 0.5:

                                    filters.append(
                                        "atempo=0.5"
                                    )

                                    remaining /= 0.5

                                filters.append(
                                    f"atempo={remaining:.6f}"
                                )

                                run_cmd(
                                    [
                                        "ffmpeg",
                                        "-y",
                                        "-i",
                                        raw_voice,
                                        "-filter:a",
                                        ",".join(
                                            filters
                                        ),
                                        "-c:a",
                                        "libmp3lame",
                                        "-q:a",
                                        "4",
                                        final_voice
                                    ]
                                )


                            actual_duration = (
                                get_media_duration(
                                    final_voice
                                )
                            )

                            scaled_durations = [
                                d / speed
                                for d in durations
                            ]

                            predicted = sum(
                                scaled_durations
                            )

                            if predicted > 0:

                                scale = (
                                    actual_duration
                                    / predicted
                                )

                                if (
                                    0.98
                                    <= scale
                                    <= 1.02
                                ):

                                    scaled_durations = [
                                        d * scale
                                        for d in scaled_durations
                                    ]


                            subtitles = []

                            current = 0.0

                            for chunk, d in zip(
                                chunks,
                                scaled_durations
                            ):

                                subtitles.append(
                                    {
                                        "start":
                                        current,

                                        "end":
                                        current + d,

                                        "text":
                                        chunk
                                    }
                                )

                                current += d


                            st.session_state[
                                "voice_path"
                            ] = final_voice

                            st.session_state[
                                "voice_duration"
                            ] = actual_duration

                            st.session_state[
                                "subtitles"
                            ] = subtitles


                        with st.spinner(
                            "🎙️ Generating Myanmar voice..."
                        ):

                            try:

                                asyncio.run(
                                    generate_tts()
                                )

                                st.success(
                                    "✅ Myanmar Female Voiceover generated!"
                                )

                            except Exception as e:

                                st.error(
                                    f"❌ TTS Error: {e}"
                                )


                    voice_path = (
                        st.session_state.get(
                            "voice_path",
                            ""
                        )
                    )

                    voice_duration = (
                        st.session_state.get(
                            "voice_duration",
                            0
                        )
                    )

                    if (
                        voice_path
                        and os.path.exists(
                            voice_path
                        )
                    ):

                        st.audio(
                            voice_path
                        )

                        st.info(
                            f"🎙️ Voiceover Duration: "
                            f"{voice_duration:.2f} seconds"
                        )


                    # =================================================
                    # SUBTITLE
                    # =================================================

                    subtitles = (
                        st.session_state.get(
                            "subtitles",
                            []
                        )
                    )

                    if subtitles:

                        st.divider()

                        st.subheader(
                            "🇲🇲 Myanmar Subtitle"
                        )

                        st.success(
                            f"✅ {len(subtitles)} subtitles "
                            f"were synced to the generated voice."
                        )

                        st.caption(
                            "Subtitle timing source: "
                            "Actual TTS audio duration"
                        )

                        st.caption(
                            "CPU optimized TTS pipeline completed."
                        )


                        # =================================================
                        # CREATE SUBTITLE FILES
                        # =================================================

                        srt_path = os.path.join(
                            temp_dir,
                            "myanmar_subtitles.srt"
                        )

                        ass_path = os.path.join(
                            temp_dir,
                            "myanmar_subtitles.ass"
                        )

                        with open(
                            srt_path,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            f.write(
                                create_srt(
                                    subtitles
                                )
                            )

                        with open(
                            ass_path,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            f.write(
                                create_ass(
                                    subtitles
                                )
                            )


                        # =================================================
                        # SCENE TIMING
                        # =================================================

                        st.divider()

                        st.subheader(
                            "⏱️ Scene Timing"
                        )

                        st.success(
                            f"Timing fixed: "
                            f"{len(subtitles)} subtitles"
                        )


                        # =================================================
                        # FREEZE + ZOOM
                        # =================================================

                        st.divider()

                        st.subheader(
                            "🧊 Freeze Frame + 🔍 Zoom"
                        )

                        enable_effects = st.checkbox(
                            "Enable Freeze Frame + Zoom",
                            value=True
                        )

                        freeze_interval = st.slider(
                            "Freeze / Zoom Interval",
                            5,
                            60,
                            10
                        )

                        freeze_duration = st.slider(
                            "Freeze Duration",
                            0.5,
                            5.0,
                            2.0,
                            0.5
                        )


                        # =================================================
                        # BLUR
                        # =================================================

                        st.divider()

                        st.subheader(
                            "🟦 Original Subtitle Blur"
                        )

                        if "blur_x" not in st.session_state:
                            st.session_state[
                                "blur_x"
                            ] = 20

                        if "blur_y" not in st.session_state:
                            st.session_state[
                                "blur_y"
                            ] = 860

                        if "blur_width" not in st.session_state:
                            st.session_state[
                                "blur_width"
                            ] = 536

                        if "blur_height" not in st.session_state:
                            st.session_state[
                                "blur_height"
                            ] = 100

                        if "blur_strength" not in st.session_state:
                            st.session_state[
                                "blur_strength"
                            ] = 15

                        if "blur_canvas_reset" not in st.session_state:
                            st.session_state[
                                "blur_canvas_reset"
                            ] = 0


                        blur_enabled = st.checkbox(
                            "🟦 Blur Mode ON / OFF",
                            value=False,
                            key="blur_enabled"
                        )


                        if blur_enabled:

                            preview_time = st.slider(
                                "Preview Time",
                                0.0,
                                max(
                                    0.1,
                                    float(duration)
                                ),
                                min(
                                    1.0,
                                    float(duration)
                                ),
                                0.1
                            )

                            preview_frame = (
                                extract_preview_frame(
                                    input_path,
                                    preview_time,
                                    576,
                                    1024
                                )
                            )


                            if preview_frame:

                                initial_drawing = (
                                    make_blur_drawing(
                                        st.session_state[
                                            "blur_x"
                                        ],
                                        st.session_state[
                                            "blur_y"
                                        ],
                                        st.session_state[
                                            "blur_width"
                                        ],
                                        st.session_state[
                                            "blur_height"
                                        ]
                                    )
                                )

                                canvas_key = (
                                    "blur_canvas_"
                                    + str(
                                        st.session_state[
                                            "blur_canvas_reset"
                                        ]
                                    )
                                )


                                st.info(
                                    "📱 Red box ကို "
                                    "ရွှေ့/ချဲ့/ချုံ့နိုင်ပါတယ်။"
                                )


                                canvas_result = st_canvas(
                                    fill_color=(
                                        "rgba("
                                        "255,0,0,0.25)"
                                    ),

                                    stroke_width=4,

                                    stroke_color="#FF0000",

                                    background_image=(
                                        preview_frame
                                    ),

                                    update_streamlit=True,

                                    height=1024,

                                    width=576,

                                    drawing_mode="rect",

                                    initial_drawing=(
                                        initial_drawing
                                    ),

                                    disabled=False,

                                    max_display_height=650,

                                    key=canvas_key
                                )


                                if (
                                    canvas_result
                                    is not None
                                    and canvas_result.json_data
                                ):

                                    objects = (
                                        canvas_result
                                        .json_data
                                        .get(
                                            "objects",
                                            []
                                        )
                                    )

                                    rects = [
                                        obj
                                        for obj in objects
                                        if obj.get(
                                            "type"
                                        )
                                        in [
                                            "rect",
                                            "Rect"
                                        ]
                                    ]

                                    if rects:

                                        obj = rects[-1]

                                        x = float(
                                            obj.get(
                                                "left",
                                                st.session_state[
                                                    "blur_x"
                                                ]
                                            )
                                        )

                                        y = float(
                                            obj.get(
                                                "top",
                                                st.session_state[
                                                    "blur_y"
                                                ]
                                            )
                                        )

                                        w = float(
                                            obj.get(
                                                "width",
                                                st.session_state[
                                                    "blur_width"
                                                ]
                                            )
                                        )

                                        h = float(
                                            obj.get(
                                                "height",
                                                st.session_state[
                                                    "blur_height"
                                                ]
                                            )
                                        )

                                        sx = float(
                                            obj.get(
                                                "scaleX",
                                                1
                                            )
                                        )

                                        sy = float(
                                            obj.get(
                                                "scaleY",
                                                1
                                            )
                                        )

                                        w *= sx
                                        h *= sy

                                        (
                                            x,
                                            y,
                                            w,
                                            h
                                        ) = normalize_blur_box(
                                            x,
                                            y,
                                            w,
                                            h
                                        )

                                        st.session_state[
                                            "blur_x"
                                        ] = x

                                        st.session_state[
                                            "blur_y"
                                        ] = y

                                        st.session_state[
                                            "blur_width"
                                        ] = w

                                        st.session_state[
                                            "blur_height"
                                        ] = h


                                if st.button(
                                    "🔄 Reset Blur Box",
                                    use_container_width=True
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


                                blur_strength = st.slider(
                                    "Blur Strength",
                                    1,
                                    30,
                                    st.session_state[
                                        "blur_strength"
                                    ]
                                )

                                st.session_state[
                                    "blur_strength"
                                ] = blur_strength

                                st.write(
                                    f"📍 X: "
                                    f"{st.session_state['blur_x']} | "
                                    f"Y: "
                                    f"{st.session_state['blur_y']}"
                                )

                                st.write(
                                    f"📐 Width: "
                                    f"{st.session_state['blur_width']} | "
                                    f"Height: "
                                    f"{st.session_state['blur_height']}"
                                )


                        # =================================================
                        # VOICEOVER TIMING
                        # =================================================

                        st.divider()

                        st.subheader(
                            "⏱️ Voiceover Timing"
                        )

                        st.info(
                            f"🎙️ Voiceover Duration: "
                            f"{voice_duration:.2f} seconds"
                        )


                        # =================================================
                        # FINAL VIDEO EXPORT
                        # =================================================

                        st.divider()

                        st.subheader(
                            "🎬 Final Video Export"
                        )


                        if st.button(
                            "🎬 Export Final Video",
                            use_container_width=True
                        ):

                            final_path = os.path.join(
                                temp_dir,
                                "final_movie_recap.mp4"
                            )

                            try:

                                # =====================================
                                # 1. NORMALIZE VIDEO
                                # =====================================

                                st.info(
                                    "🎬 Preparing video..."
                                )

                                normalized_video = os.path.join(
                                    temp_dir,
                                    "normalized_video.mp4"
                                )

                                normalize_cmd = [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    input_path,
                                    "-vf",
                                    (
                                        "scale=576:1024:"
                                        "force_original_aspect_ratio=decrease,"
                                        "pad=576:1024:"
                                        "(ow-iw)/2:"
                                        "(oh-ih)/2,"
                                        "setsar=1,"
                                        "fps=30"
                                    ),
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "ultrafast",
                                    "-crf",
                                    "27",
                                    normalized_video
                                ]

                                result = subprocess.run(
                                    normalize_cmd,
                                    capture_output=True,
                                    text=True
                                )

                                if result.returncode != 0:

                                    raise RuntimeError(
                                        "Video preparation failed:\n"
                                        + result.stderr[-4000:]
                                    )


                                # =====================================
                                # 2. ZOOM
                                # =====================================

                                base_video = (
                                    normalized_video
                                )

                                if enable_effects:

                                    st.info(
                                        "🔍 Applying Zoom..."
                                    )

                                    zoom_video = os.path.join(
                                        temp_dir,
                                        "zoom_video.mp4"
                                    )

                                    zoom_filter = (
                                        "zoompan="
                                        "z='min(zoom+0.0015,1.15)':"
                                        "x='iw/2-(iw/zoom/2)':"
                                        "y='ih/2-(ih/zoom/2)':"
                                        "d=1:"
                                        "s=576x1024:"
                                        "fps=30"
                                    )

                                    zoom_cmd = [
                                        "ffmpeg",
                                        "-y",
                                        "-i",
                                        normalized_video,
                                        "-vf",
                                        zoom_filter,
                                        "-an",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "ultrafast",
                                        "-crf",
                                        "27",
                                        zoom_video
                                    ]

                                    result = subprocess.run(
                                        zoom_cmd,
                                        capture_output=True,
                                        text=True
                                    )

                                    if result.returncode == 0:

                                        base_video = (
                                            zoom_video
                                        )

                                    else:

                                        st.warning(
                                            "⚠️ Zoom failed. "
                                            "Using normal video."
                                        )


                                # =====================================
                                # 3. BLUR
                                # =====================================

                                video_after_blur = (
                                    base_video
                                )

                                if blur_enabled:

                                    st.info(
                                        "🟦 Applying original "
                                        "subtitle blur..."
                                    )

                                    blurred_video = os.path.join(
                                        temp_dir,
                                        "blurred_video.mp4"
                                    )

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

                                    bx = max(
                                        0,
                                        min(
                                            bx,
                                            575
                                        )
                                    )

                                    by = max(
                                        0,
                                        min(
                                            by,
                                            1023
                                        )
                                    )

                                    bw = max(
                                        10,
                                        min(
                                            bw,
                                            576 - bx
                                        )
                                    )

                                    bh = max(
                                        10,
                                        min(
                                            bh,
                                            1024 - by
                                        )
                                    )

                                    blur_filter = (
                                        "[0:v]"
                                        "split=2"
                                        "[main][blur];"

                                        "[blur]"
                                        f"crop={bw}:{bh}:{bx}:{by},"
                                        f"boxblur="
                                        f"luma_radius={br}:"
                                        f"luma_power=2"
                                        "[blurregion];"

                                        "[main]"
                                        "[blurregion]"
                                        f"overlay={bx}:{by}"
                                        "[outv]"
                                    )

                                    blur_cmd = [
                                        "ffmpeg",
                                        "-y",
                                        "-i",
                                        base_video,
                                        "-filter_complex",
                                        blur_filter,
                                        "-map",
                                        "[outv]",
                                        "-an",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "ultrafast",
                                        "-crf",
                                        "27",
                                        blurred_video
                                    ]

                                    result = subprocess.run(
                                        blur_cmd,
                                        capture_output=True,
                                        text=True
                                    )

                                    if result.returncode != 0:

                                        raise RuntimeError(
                                            "Blur failed:\n"
                                            + result.stderr[-4000:]
                                        )

                                    video_after_blur = (
                                        blurred_video
                                    )


                                # =====================================
                                # 4. MYANMAR SUBTITLE
                                # =====================================

                                st.info(
                                    "🇲🇲 Adding Myanmar subtitles..."
                                )

                                subtitled_video = os.path.join(
                                    temp_dir,
                                    "subtitled_video.mp4"
                                )

                                subtitle_cmd = [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    video_after_blur,
                                    "-vf",
                                    f"ass={ass_path}",
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "ultrafast",
                                    "-crf",
                                    "27",
                                    subtitled_video
                                ]

                                result = subprocess.run(
                                    subtitle_cmd,
                                    capture_output=True,
                                    text=True
                                )

                                if result.returncode != 0:

                                    raise RuntimeError(
                                        "Subtitle failed:\n"
                                        + result.stderr[-4000:]
                                    )


                                # =====================================
                                # 5. DURATION
                                # =====================================

                                video_duration = (
                                    get_media_duration(
                                        subtitled_video
                                    )
                                )

                                voice_final_duration = (
                                    get_media_duration(
                                        voice_path
                                    )
                                )


                                # =====================================
                                # 6. EXTEND VIDEO
                                # =====================================

                                if (
                                    voice_final_duration
                                    > video_duration
                                ):

                                    st.info(
                                        "⏱️ Extending video..."
                                    )

                                    extended_video = os.path.join(
                                        temp_dir,
                                        "extended_video.mp4"
                                    )

                                    extra = (
                                        voice_final_duration
                                        - video_duration
                                    )

                                    extend_cmd = [
                                        "ffmpeg",
                                        "-y",
                                        "-i",
                                        subtitled_video,
                                        "-vf",
                                        (
                                            "tpad="
                                            "stop_mode=clone:"
                                            f"stop_duration={extra:.3f}"
                                        ),
                                        "-an",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "ultrafast",
                                        "-crf",
                                        "27",
                                        extended_video
                                    ]

                                    result = subprocess.run(
                                        extend_cmd,
                                        capture_output=True,
                                        text=True
                                    )

                                    if result.returncode != 0:

                                        raise RuntimeError(
                                            "Video extension failed:\n"
                                            + result.stderr[-4000:]
                                        )

                                    subtitled_video = (
                                        extended_video
                                    )


                                # =====================================
                                # 7. ADD VOICE
                                # =====================================

                                st.info(
                                    "🎙️ Adding Myanmar voiceover..."
                                )

                                final_cmd = [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    subtitled_video,
                                    "-i",
                                    voice_path,

                                    "-map",
                                    "0:v:0",
                                    "-map",
                                    "1:a:0",

                                    "-c:v",
                                    "copy",

                                    "-c:a",
                                    "aac",

                                    "-b:a",
                                    "128k",

                                    "-shortest",

                                    final_path
                                ]

                                result = subprocess.run(
                                    final_cmd,
                                    capture_output=True,
                                    text=True
                                )

                                if result.returncode != 0:

                                    raise RuntimeError(
                                        "Final video creation failed:\n"
                                        + result.stderr[-5000:]
                                    )


                                # =====================================
                                # 8. SUCCESS
                                # =====================================

                                if not os.path.exists(
                                    final_path
                                ):

                                    raise RuntimeError(
                                        "Final MP4 file was not created."
                                    )

                                file_size = (
                                    os.path.getsize(
                                        final_path
                                    )
                                    / (1024 * 1024)
                                )

                                st.success(
                                    "🎉 Final video export completed!"
                                )

                                st.info(
                                    f"📦 Final video: "
                                    f"{file_size:.2f} MB"
                                )

                                st.video(
                                    final_path
                                )

                                with open(
                                    final_path,
                                    "rb"
                                ) as f:

                                    st.download_button(
                                        "⬇️ Download Final Video",
                                        f,
                                        file_name=(
                                            "final_movie_recap.mp4"
                                        ),
                                        mime="video/mp4",
                                        use_container_width=True
                                    )


                            except Exception as e:

                                st.error(
                                    "❌ Export Error"
                                )

                                st.code(
                                    str(e)
                                )

    else:

        st.error(
            "❌ Cannot open uploaded video."
        )
