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
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recap AI")
st.write("Upload a movie and analyze video information.")


# =========================================================
# HELPERS
# =========================================================

def get_audio_duration(audio_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ],
            capture_output=True,
            text=True
        )

        return float(result.stdout.strip())

    except Exception:
        return 0.0


def ass_time(seconds):
    seconds = max(0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)

    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def split_myanmar_text(text, max_chars=65):
    """
    Split Myanmar recap into subtitle chunks.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[။！？!?])\s*",
        text
    )

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(current) + len(sentence) <= max_chars:
            current = (
                sentence
                if not current
                else current + " " + sentence
            )

        else:

            if current:
                chunks.append(current)

            current = sentence

    if current:
        chunks.append(current)

    return chunks


def wrap_myanmar(text, max_chars=24):
    """
    Wrap subtitle text into up to 3 lines.
    """

    text = text.strip()

    if len(text) <= max_chars:
        return text

    words = text.split()

    lines = []
    current = ""

    for word in words:

        test = word if not current else current + " " + word

        if len(test) <= max_chars:
            current = test

        else:

            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    if len(lines) <= 3:
        return r"\N".join(lines)

    # If too many lines, distribute into 3 lines
    total = len(text)
    part = max(1, total // 3)

    line1 = text[:part]
    line2 = text[part:part * 2]
    line3 = text[part * 2:]

    return (
        line1.strip()
        + r"\N"
        + line2.strip()
        + r"\N"
        + line3.strip()
    )


# =========================================================
# VIDEO FRAME EXTRACTION
# =========================================================

def extract_frame_at_timestamp(cap, timestamp):
    """
    Extract one representative frame at a specific timestamp.
    """

    try:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            max(0, float(timestamp)) * 1000
        )

        ret, frame = cap.read()

        if not ret or frame is None:
            return None

        # Keep original aspect ratio.
        # Resize only if image is too large.
        max_side = 768

        h, w = frame.shape[:2]

        scale = min(
            1.0,
            max_side / max(h, w)
        )

        if scale < 1.0:

            new_w = int(w * scale)
            new_h = int(h * scale)

            frame = cv2.resize(
                frame,
                (new_w, new_h),
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


def extract_visual_frames_for_segments(
    video_path,
    segments,
    max_frames=24
):
    """
    Extract one frame from the middle of each Whisper segment.

    If there are too many segments, sample them evenly.
    """

    if not segments:
        return []

    # Clean usable segments
    usable = []

    for i, seg in enumerate(segments):

        try:

            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))

            text = str(
                seg.get("text", "")
            ).strip()

            if end <= start:
                continue

            usable.append(
                {
                    "original_index": i,
                    "start": start,
                    "end": end,
                    "text": text
                }
            )

        except Exception:
            continue

    if not usable:
        return []

    # If there are too many segments,
    # choose evenly distributed segments.
    if len(usable) > max_frames:

        indexes = []

        for i in range(max_frames):

            position = (
                i * (len(usable) - 1)
                / (max_frames - 1)
            )

            indexes.append(
                round(position)
            )

        selected = [
            usable[i]
            for i in indexes
        ]

    else:

        selected = usable

    frames = []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    try:

        for item_number, seg in enumerate(selected, start=1):

            midpoint = (
                seg["start"] +
                seg["end"]
            ) / 2

            frame_bytes = extract_frame_at_timestamp(
                cap,
                midpoint
            )

            if frame_bytes:

                frames.append(
                    {
                        "number": item_number,
                        "start": seg["start"],
                        "end": seg["end"],
                        "timestamp": midpoint,
                        "text": seg["text"],
                        "image": frame_bytes
                    }
                )

    finally:

        cap.release()

    return frames


# =========================================================
# WHISPER
# =========================================================

@st.cache_resource
def load_whisper_model(model_name):

    import whisper

    return whisper.load_model(model_name)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "transcript": "",
    "whisper_segments": [],
    "ai_scene_analysis": "",
    "movie_recap": "",
    "myanmar_recap": "",
    "voiceover_path": "",
    "subtitle_timing": [],
    "subtitle_timing_source": "",
    "visual_frames_checked": 0
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.header("⚙️ Settings")


whisper_model_name = st.sidebar.selectbox(
    "Whisper Model",
    [
        "tiny",
        "base"
    ],
    index=1
)


voice_gender = st.sidebar.selectbox(
    "Myanmar Voice",
    [
        "Female",
        "Male"
    ]
)


voice_speed = st.sidebar.selectbox(
    "Voice Speed",
    [
        1.0,
        1.1,
        1.2
    ],
    index=0
)


st.sidebar.markdown("---")


st.sidebar.subheader("🧊 Freeze Frame")


freeze_enabled = st.sidebar.checkbox(
    "Enable Freeze Frame",
    value=True
)


freeze_interval = st.sidebar.slider(
    "Freeze Interval",
    min_value=5,
    max_value=60,
    value=10,
    step=1
)


freeze_duration = st.sidebar.slider(
    "Maximum Freeze Duration",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
)


# =========================================================
# ZOOM SETTINGS
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Zoom")


zoom_enabled = st.sidebar.checkbox(
    "Enable Zoom",
    value=True
)


zoom_level = st.sidebar.slider(
    "Zoom Level",
    min_value=1.0,
    max_value=1.5,
    value=1.1,
    step=0.1
)


zoom_duration = st.sidebar.slider(
    "Zoom Duration",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
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


if uploaded_file:

    file_size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.info(
        f"📦 File Size: {file_size_mb:.2f} MB"
    )

    if file_size_mb > 200:

        st.error(
            "❌ Maximum file size is 200MB."
        )

        st.stop()


    # -----------------------------------------------------
    # SAVE TEMP VIDEO
    # -----------------------------------------------------

    temp_dir = tempfile.mkdtemp()

    video_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(video_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # -----------------------------------------------------
    # VIDEO INFORMATION
    # -----------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():

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

        video_duration = (
            frame_count / fps
            if fps > 0
            else 0
        )

        cap.release()

    else:

        fps = 0
        frame_count = 0
        width = 0
        height = 0
        video_duration = 0


    st.subheader("🎬 Video Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Duration",
            f"{video_duration:.1f} sec"
        )

    with col2:
        st.metric(
            "Resolution",
            f"{width}×{height}"
        )

    with col3:
        st.metric(
            "FPS",
            f"{fps:.0f}"
        )

    with col4:
        st.metric(
            "File Size",
            f"{file_size_mb:.2f} MB"
        )


    st.video(video_path)


    # =====================================================
    # API CLIENT
    # =====================================================

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        ""
    )


    if not api_key:

        st.error(
            "❌ GEMINI_API_KEY is not configured."
        )

        st.stop()


    client = genai.Client(
        api_key=api_key
    )


    # =====================================================
    # 1. TRANSCRIPT
    # =====================================================

    st.header("📝 Transcript")

    if st.button(
        "🎙️ Generate Transcript",
        use_container_width=True
    ):

        with st.spinner(
            "Transcribing video..."
        ):

            try:

                model = load_whisper_model(
                    whisper_model_name
                )

                result = model.transcribe(
                    video_path,
                    language="en",
                    fp16=False
                )

                transcript = result.get(
                    "text",
                    ""
                ).strip()

                segments = result.get(
                    "segments",
                    []
                )

                st.session_state[
                    "transcript"
                ] = transcript

                st.session_state[
                    "whisper_segments"
                ] = segments

            except Exception as e:

                st.error(
                    f"❌ Whisper Error: {e}"
                )


    if st.session_state["transcript"]:

        st.text_area(
            "Transcript",
            st.session_state["transcript"],
            height=250
        )


    # =====================================================
    # 2. SCENE ANALYSIS
    # =====================================================

    st.header("🎞️ Scene Analysis")


    if st.session_state["whisper_segments"]:

        st.write(
            f"Detected "
            f"{len(st.session_state['whisper_segments'])} "
            f"timestamped transcript segments."
        )


        for i, seg in enumerate(
            st.session_state["whisper_segments"],
            start=1
        ):

            start = float(
                seg.get("start", 0)
            )

            end = float(
                seg.get("end", 0)
            )

            text = seg.get(
                "text",
                ""
            ).strip()

            st.markdown(
                f"**{i}. "
                f"{start:.2f}s → {end:.2f}s**"
            )

            st.write(text)


    # =====================================================
    # 3. AI VISUAL SCENE ANALYSIS
    # =====================================================

    st.header("🤖 AI Scene Analysis")

    if st.button(
        "🔍 Analyze Actual Video Scenes",
        use_container_width=True
    ):

        if not st.session_state[
            "whisper_segments"
        ]:

            st.warning(
                "Please generate Transcript first."
            )

        else:

            with st.spinner(
                "🎬 Extracting video frames and "
                "checking scenes with transcript..."
            ):

                try:

                    segments = st.session_state[
                        "whisper_segments"
                    ]

                    # -------------------------------------------------
                    # IMPORTANT:
                    # Take actual frames from the video at the
                    # middle of Whisper timestamp segments.
                    # -------------------------------------------------

                    visual_frames = (
                        extract_visual_frames_for_segments(
                            video_path,
                            segments,
                            max_frames=24
                        )
                    )


                    st.session_state[
                        "visual_frames_checked"
                    ] = len(visual_frames)


                    if not visual_frames:

                        st.error(
                            "❌ Could not extract video frames."
                        )

                    else:

                        # -------------------------------------------------
                        # BUILD MULTIMODAL GEMINI CONTENT
                        # -------------------------------------------------

                        prompt = """
You are a VISUAL-FIRST movie scene analyst.

You will receive:
1. Timestamped Whisper transcript segments.
2. One actual video frame captured from the middle of each
   selected transcript segment.

Your job is to create a VERIFIED scene analysis that matches
the actual video.

VERY IMPORTANT RULES:

1. The ACTUAL VIDEO FRAME is the source of truth for visual events.
2. Do NOT invent people, characters, locations, objects, actions,
   emotions, events, or story details that are not visible or
   clearly supported.
3. Use the transcript only to understand spoken dialogue or narration.
4. Keep the exact chronological order of timestamps.
5. NEVER move dialogue from one timestamp to another.
6. NEVER change the order of scenes.
7. If the transcript says something but the frame does not visually
   confirm it, do not turn that statement into a visual event.
8. If visual information and transcript information appear different,
   keep them separate instead of forcing them to match.
9. Do not guess hidden actions outside the visible frame.
10. Do not create events that are not present.
11. Each analyzed item must reference its original timestamp.
12. The final movie recap will be generated ONLY from your verified
    scene analysis, so accuracy is more important than creativity.

For every item, provide:

Scene Number:
Timestamp:
Transcript:
What is visibly happening:
Visible people/characters:
Visible location/environment:
Visible objects:
Action:
Verified scene summary:

Keep the timestamp and chronological order exactly as provided.

Do not write a fictional story.
Do not add information from general movie knowledge.
Only describe what can be verified from the supplied video frames
and transcript.
"""


                        parts = []

                        parts.append(
                            types.Part.from_text(
                                text=prompt
                            )
                        )


                        # -------------------------------------------------
                        # ADD FRAMES + THEIR EXACT TIMESTAMPS
                        # -------------------------------------------------

                        for item in visual_frames:

                            label = f"""
==================================================
VIDEO FRAME {item['number']}
Timestamp: {item['start']:.2f}s - {item['end']:.2f}s
Frame captured at: {item['timestamp']:.2f}s

Whisper transcript for this timestamp:
{item['text']}

Analyze THIS frame against THIS timestamp.
==================================================
"""

                            parts.append(
                                types.Part.from_text(
                                    text=label
                                )
                            )

                            parts.append(
                                types.Part.from_bytes(
                                    data=item["image"],
                                    mime_type="image/jpeg"
                                )
                            )


                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=parts
                        )


                        analysis_text = (
                            response.text
                            if response
                            else ""
                        )


                        if analysis_text:

                            st.session_state[
                                "ai_scene_analysis"
                            ] = analysis_text

                            st.success(
                                f"✅ Verified "
                                f"{len(visual_frames)} "
                                f"video frames against transcript."
                            )

                        else:

                            st.error(
                                "❌ Gemini returned no scene analysis."
                            )


                except Exception as e:

                    st.error(
                        f"❌ AI Scene Analysis Error: {e}"
                    )


    if st.session_state[
        "ai_scene_analysis"
    ]:

        st.success(
            f"🎥 Actual video frames checked: "
            f"{st.session_state['visual_frames_checked']}"
        )

        st.text_area(
            "Verified AI Scene Analysis",
            st.session_state[
                "ai_scene_analysis"
            ],
            height=500
        )


    # =====================================================
    # 4. MOVIE RECAP SCRIPT
    # =====================================================

    st.header("🎬 Movie Recap Script")


    if st.button(
        "✍️ Generate Movie Recap Script",
        use_container_width=True
    ):

        if not st.session_state[
            "ai_scene_analysis"
        ]:

            st.warning(
                "Please run AI Scene Analysis first."
            )

        else:

            with st.spinner(
                "Writing recap from verified video scenes..."
            ):

                try:

                    recap_prompt = f"""
You are writing a movie recap narration.

IMPORTANT:
The scene analysis below has already been checked against
ACTUAL VIDEO FRAMES and timestamped transcript segments.

Use ONLY the verified scene analysis.

STRICT RULES:

1. Follow the exact chronological order.
2. Do not reorder scenes.
3. Do not invent any event.
4. Do not invent characters.
5. Do not invent locations.
6. Do not invent actions.
7. Do not add information that is not in the verified analysis.
8. Do not turn uncertain visual information into facts.
9. Do not move dialogue or events to another scene.
10. If something is unclear, leave it out.
11. Keep the recap synchronized with what actually happens
    in the supplied video.
12. Focus on important visible story events.
13. Remove unnecessary repetition.
14. Make the narration natural and engaging.
15. Write suitable narration for voiceover.
16. The recap must describe the supplied video, not a guessed
    version of the movie.
17. Do not use outside movie knowledge.
18. Do not create an ending that is not present in the analysis.

Write one continuous English movie recap narration.

Do not add:
- Scene headings
- Character explanations not present in the analysis
- Fake dialogue
- Extra story details
- Events from outside the supplied video

VERIFIED VIDEO SCENE ANALYSIS:
--------------------------------

{st.session_state["ai_scene_analysis"]}

--------------------------------

Now write the accurate chronological movie recap.
"""


                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=recap_prompt
                    )


                    recap = (
                        response.text
                        if response
                        else ""
                    )


                    if recap:

                        st.session_state[
                            "movie_recap"
                        ] = recap

                        st.success(
                            "✅ Movie Recap Script generated "
                            "from verified video scenes."
                        )

                    else:

                        st.error(
                            "❌ Gemini returned no recap."
                        )


                except Exception as e:

                    st.error(
                        f"❌ Movie Recap Error: {e}"
                    )


    if st.session_state[
        "movie_recap"
    ]:

        st.text_area(
            "Movie Recap Script",
            st.session_state[
                "movie_recap"
            ],
            height=400
        )


    # =====================================================
    # 5. MYANMAR RECAP
    # =====================================================

    st.header("🇲🇲 Myanmar Recap")


    if st.button(
        "🇲🇲 Translate to Myanmar",
        use_container_width=True
    ):

        if not st.session_state[
            "movie_recap"
        ]:

            st.warning(
                "Please generate Movie Recap Script first."
            )

        else:

            with st.spinner(
                "Translating recap into Myanmar..."
            ):

                try:

                    myanmar_prompt = f"""
Translate the following movie recap into natural Myanmar.

IMPORTANT:

- Keep the exact story meaning.
- Keep the exact chronological order.
- Do not add events.
- Do not remove important events.
- Do not invent dialogue.
- Make it natural for Myanmar voiceover.
- Keep sentences clear and easy to understand.
- Do not translate names unnecessarily.

The final Myanmar narration must describe exactly
the same events as the English recap.

English Movie Recap:

{st.session_state["movie_recap"]}
"""


                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=myanmar_prompt
                    )


                    myanmar_text = (
                        response.text
                        if response
                        else ""
                    )


                    # User's preferred ending style
                    myanmar_text = myanmar_text.replace(
                        "တယ်",
                        "ဒယ်"
                    )


                    st.session_state[
                        "myanmar_recap"
                    ] = myanmar_text


                except Exception as e:

                    st.error(
                        f"❌ Myanmar Translation Error: {e}"
                    )


    if st.session_state[
        "myanmar_recap"
    ]:

        st.text_area(
            "Myanmar Recap Script",
            st.session_state[
                "myanmar_recap"
            ],
            height=400
        )


    # =====================================================
    # 6. MYANMAR FEMALE / MALE VOICEOVER
    # =====================================================

    st.header(
        "🎙️ Myanmar Voiceover"
    )


    if st.button(
        "🎙️ Generate Myanmar Voiceover",
        use_container_width=True
    ):

        if not st.session_state[
            "myanmar_recap"
        ]:

            st.warning(
                "Please generate Myanmar Recap first."
            )

        else:

            with st.spinner(
                "Generating Myanmar voiceover..."
            ):

                try:

                    import edge_tts


                    if voice_gender == "Female":

                        voice = (
                            "my-MM-NilarNeural"
                        )

                    else:

                        voice = (
                            "my-MM-ThihaNeural"
                        )


                    # -------------------------------------------------
                    # Split narration into chunks
                    # -------------------------------------------------

                    chunks = split_myanmar_text(
                        st.session_state[
                            "myanmar_recap"
                        ],
                        max_chars=65
                    )


                    if not chunks:

                        st.error(
                            "❌ No narration text found."
                        )

                    else:

                        tts_dir = os.path.join(
                            temp_dir,
                            "tts_chunks"
                        )

                        os.makedirs(
                            tts_dir,
                            exist_ok=True
                        )


                        chunk_files = []


                        # -------------------------------------------------
                        # TTS
                        # -------------------------------------------------

                        for i, chunk in enumerate(
                            chunks
                        ):

                            output_file = os.path.join(
                                tts_dir,
                                f"chunk_{i:03d}.mp3"
                            )


                            # Edge TTS rate
                            rate_percent = int(
                                (voice_speed - 1.0)
                                * 100
                            )


                            if rate_percent >= 0:

                                rate = (
                                    f"+{rate_percent}%"
                                )

                            else:

                                rate = (
                                    f"{rate_percent}%"
                                )


                            async def generate_tts(
                                text,
                                output,
                                voice_name,
                                rate_value
                            ):

                                communicate = (
                                    edge_tts.Communicate(
                                        text,
                                        voice_name,
                                        rate=rate_value
                                    )
                                )

                                await communicate.save(
                                    output
                                )


                            asyncio.run(
                                generate_tts(
                                    chunk,
                                    output_file,
                                    voice,
                                    rate
                                )
                            )


                            if os.path.exists(
                                output_file
                            ):

                                chunk_files.append(
                                    output_file
                                )


                        if chunk_files:

                            # =================================================
                            # CROSSFADE VOICE CHUNKS
                            # =================================================

                            TTS_CROSSFADE = 0.12

                            normalized_files = []


                            for i, file_path in enumerate(
                                chunk_files
                            ):

                                normalized = os.path.join(
                                    tts_dir,
                                    f"norm_{i:03d}.wav"
                                )


                                normalize_cmd = [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    file_path,
                                    "-af",
                                    (
                                        "aformat="
                                        "sample_fmts=fltp:"
                                        "sample_rates=48000:"
                                        "channel_layouts=stereo"
                                    ),
                                    normalized
                                ]


                                subprocess.run(
                                    normalize_cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )


                                if os.path.exists(
                                    normalized
                                ):

                                    normalized_files.append(
                                        normalized
                                    )


                            voiceover_path = os.path.join(
                                temp_dir,
                                "myanmar_voiceover.mp3"
                            )


                            if len(normalized_files) == 1:

                                shutil.copy(
                                    normalized_files[0],
                                    voiceover_path
                                )

                            else:

                                # -----------------------------------------
                                # Build acrossfade filter
                                # -----------------------------------------

                                inputs = []

                                for i in range(
                                    len(normalized_files)
                                ):

                                    inputs.extend(
                                        [
                                            "-i",
                                            normalized_files[i]
                                        ]
                                    )


                                filter_parts = []

                                previous = "[0:a]"


                                for i in range(
                                    1,
                                    len(normalized_files)
                                ):

                                    output_label = (
                                        f"[a{i}]"
                                    )

                                    filter_parts.append(
                                        (
                                            f"{previous}"
                                            f"[{i}:a]"
                                            f"acrossfade="
                                            f"d={TTS_CROSSFADE}:"
                                            f"c1=tri:c2=tri"
                                            f"{output_label}"
                                        )
                                    )

                                    previous = output_label


                                filter_complex = ";".join(
                                    filter_parts
                                )


                                combine_cmd = [
                                    "ffmpeg",
                                    "-y"
                                ]

                                combine_cmd.extend(
                                    inputs
                                )

                                combine_cmd.extend(
                                    [
                                        "-filter_complex",
                                        filter_complex,
                                        "-map",
                                        previous,
                                        "-c:a",
                                        "libmp3lame",
                                        "-b:a",
                                        "192k",
                                        voiceover_path
                                    ]
                                )


                                subprocess.run(
                                    combine_cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )


                            if os.path.exists(
                                voiceover_path
                            ):

                                st.session_state[
                                    "voiceover_path"
                                ] = voiceover_path


                                # -----------------------------------------
                                # Calculate subtitle timing from TTS chunks
                                # -----------------------------------------

                                subtitle_timing = []

                                current_time = 0.0


                                for i, chunk_file in enumerate(
                                    chunk_files
                                ):

                                    duration = (
                                        get_audio_duration(
                                            chunk_file
                                        )
                                    )


                                    start_time = current_time

                                    end_time = (
                                        start_time
                                        + duration
                                    )


                                    subtitle_timing.append(
                                        {
                                            "start": start_time,
                                            "end": end_time,
                                            "text": chunks[i]
                                        }
                                    )


                                    current_time = (
                                        end_time
                                        - TTS_CROSSFADE
                                    )


                                st.session_state[
                                    "subtitle_timing"
                                ] = subtitle_timing


                                st.session_state[
                                    "subtitle_timing_source"
                                ] = (
                                    "tts_segments_crossfade"
                                )


                                st.success(
                                    "✅ Myanmar voiceover generated."
                                )


                                st.audio(
                                    voiceover_path
                                )


                except Exception as e:

                    st.error(
                        f"❌ Voiceover Error: {e}"
                    )


    # =====================================================
    # 7. MYANMAR SUBTITLE
    # =====================================================

    st.header(
        "🇲🇲 Myanmar Subtitle"
    )


    if st.session_state[
        "myanmar_recap"
    ]:

        subtitle_text = st.session_state[
            "myanmar_recap"
        ]

        subtitle_chunks = split_myanmar_text(
            subtitle_text,
            max_chars=65
        )


        if st.session_state[
            "subtitle_timing"
        ]:

            timings = st.session_state[
                "subtitle_timing"
            ]

            subtitle_items = []

            for i, item in enumerate(
                timings
            ):

                if i >= len(
                    subtitle_chunks
                ):
                    break

                subtitle_items.append(
                    {
                        "start": item["start"],
                        "end": item["end"],
                        "text": subtitle_chunks[i]
                    }
                )


        else:

            # -------------------------------------------------
            # Fallback timing based on video duration
            # -------------------------------------------------

            subtitle_items = []

            if subtitle_chunks:

                chunk_duration = (
                    video_duration /
                    len(subtitle_chunks)
                )

                for i, text in enumerate(
                    subtitle_chunks
                ):

                    subtitle_items.append(
                        {
                            "start":
                                i * chunk_duration,
                            "end":
                                (i + 1)
                                * chunk_duration,
                            "text": text
                        }
                    )


        st.session_state[
            "subtitle_timing"
        ] = subtitle_items


        # =================================================
        # SRT
        # =================================================

        srt_path = os.path.join(
            temp_dir,
            "myanmar_subtitles.srt"
        )


        with open(
            srt_path,
            "w",
            encoding="utf-8"
        ) as f:

            for i, item in enumerate(
                subtitle_items,
                start=1
            ):

                start = item["start"]
                end = item["end"]
                text = item["text"]


                def srt_time(sec):

                    sec = max(
                        0,
                        float(sec)
                    )

                    hours = int(
                        sec // 3600
                    )

                    minutes = int(
                        (sec % 3600) // 60
                    )

                    seconds = int(
                        sec % 60
                    )

                    millis = int(
                        (sec - int(sec))
                        * 1000
                    )

                    return (
                        f"{hours:02d}:"
                        f"{minutes:02d}:"
                        f"{seconds:02d},"
                        f"{millis:03d}"
                    )


                f.write(
                    f"{i}\n"
                    f"{srt_time(start)} --> "
                    f"{srt_time(end)}\n"
                    f"{text}\n\n"
                )


        # =================================================
        # ASS
        # =================================================

        ass_path = os.path.join(
            temp_dir,
            "myanmar_subtitles.ass"
        )


        with open(
            ass_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "[Script Info]\n"
            )

            f.write(
                "ScriptType: v4.00+\n"
            )

            f.write(
                "PlayResX: 576\n"
            )

            f.write(
                "PlayResY: 1024\n"
            )

            f.write(
                "ScaledBorderAndShadow: yes\n\n"
            )


            f.write(
                "[V4+ Styles]\n"
            )

            f.write(
                "Format: Name, Fontname, Fontsize, "
                "PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, Alignment, "
                "MarginL, MarginR, MarginV, Encoding\n"
            )


            f.write(
                "Style: Myanmar,"
                "Noto Sans Myanmar,"
                "28,"
                "&H00FFFFFF,"
                "&H00FFFFFF,"
                "&H00000000,"
                "&H80000000,"
                "0,0,0,0,"
                "100,100,0,0,"
                "1,2,1,2,"
                "30,30,70,1\n\n"
            )


            f.write(
                "[Events]\n"
            )

            f.write(
                "Format: Layer, Start, End, Style, "
                "Name, MarginL, MarginR, MarginV, "
                "Effect, Text\n"
            )


            for item in subtitle_items:

                start = ass_time(
                    item["start"]
                )

                end = ass_time(
                    item["end"]
                )

                text = wrap_myanmar(
                    item["text"],
                    max_chars=24
                )


                # Escape ASS characters
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


                f.write(
                    f"Dialogue: 0,"
                    f"{start},"
                    f"{end},"
                    f"Myanmar,"
                    f",0,0,0,,"
                    f"{text}\n"
                )


        st.success(
            f"✅ Timing fixed: "
            f"{len(subtitle_items)} subtitles"
        )


        # Preview subtitle data

        with st.expander(
            "📄 Subtitle Timing Preview"
        ):

            for i, item in enumerate(
                subtitle_items,
                start=1
            ):

                st.write(
                    f"{i}. "
                    f"{item['start']:.2f}s → "
                    f"{item['end']:.2f}s"
                )

                st.write(
                    item["text"]
                )


    # =====================================================
    # 8. SUBTITLE EXPORT
    # =====================================================

    st.header(
        "📤 Subtitle Export"
    )


    if os.path.exists(
        srt_path
    ):

        with open(
            srt_path,
            "rb"
        ) as f:

            st.download_button(
                "⬇️ Download SRT",
                f,
                file_name="myanmar_subtitles.srt",
                mime="application/x-subrip"
            )


    if os.path.exists(
        ass_path
    ):

        with open(
            ass_path,
            "rb"
        ) as f:

            st.download_button(
                "⬇️ Download ASS",
                f,
                file_name="myanmar_subtitles.ass",
                mime="text/plain"
            )


    # =====================================================
    # 9. SCENE TIMING
    # =====================================================

    st.header(
        "⏱️ Scene Timing"
    )


    if st.session_state[
        "subtitle_timing"
    ]:

        st.success(
            f"Timing fixed: "
            f"{len(st.session_state['subtitle_timing'])} "
            f"subtitles"
        )


    # =====================================================
    # 10. FREEZE FRAME
    # =====================================================

    st.header(
        "🧊 Freeze Frame"
    )


    st.write(
        f"Enable Freeze Frame: "
        f"{'ON' if freeze_enabled else 'OFF'}"
    )

    st.write(
        f"Freeze every {freeze_interval} seconds"
    )

    st.write(
        f"Freeze duration: "
        f"{freeze_duration:.2f} seconds"
    )


    # =====================================================
    # 11. ZOOM
    # =====================================================

    st.header(
        "🔍 Zoom In / Zoom Out"
    )


    st.write(
        f"Enable Zoom: "
        f"{'ON' if zoom_enabled else 'OFF'}"
    )

    st.write(
        f"Zoom Level: {zoom_level:.1f}x"
    )

    st.write(
        f"Zoom Duration: "
        f"{zoom_duration:.2f} seconds"
    )


    # =====================================================
    # 12. VOICEOVER TIMING
    # =====================================================

    st.header(
        "⏱️ Voiceover Timing"
    )


    voice_duration = 0.0


    if st.session_state[
        "voiceover_path"
    ] and os.path.exists(
        st.session_state[
            "voiceover_path"
        ]
    ):

        voice_duration = (
            get_audio_duration(
                st.session_state[
                    "voiceover_path"
                ]
            )
        )


        st.write(
            f"🎙️ Voiceover Duration: "
            f"{voice_duration:.1f} seconds"
        )


    # =====================================================
    # 13. FINAL VIDEO EXPORT
    # =====================================================

    st.header(
        "🎬 Final Video Export"
    )


    if st.button(
        "🚀 Export Final Video",
        use_container_width=True
    ):

        if not st.session_state[
            "voiceover_path"
        ]:

            st.warning(
                "Please generate voiceover first."
            )

        else:

            with st.spinner(
                "🎬 Creating final video..."
            ):

                try:

                    original_video = os.path.join(
                        temp_dir,
                        "movie_recap_original.mp4"
                    )


                    shutil.copy(
                        video_path,
                        original_video
                    )


                    # =================================================
                    # VIDEO PROCESSING
                    # =================================================

                    processed_video = os.path.join(
                        temp_dir,
                        "processed_video.mp4"
                    )


                    # -------------------------------------------------
                    # Freeze Frame
                    # -------------------------------------------------

                    if freeze_enabled:

                        freeze_parts = []

                        current = 0.0
                        part_index = 0


                        while current < video_duration:

                            segment_end = min(
                                current
                                + freeze_interval,
                                video_duration
                            )


                            normal_file = os.path.join(
                                temp_dir,
                                f"normal_{part_index}.mp4"
                            )


                            duration = (
                                segment_end
                                - current
                            )


                            normal_cmd = [
                                "ffmpeg",
                                "-y",
                                "-ss",
                                str(current),
                                "-i",
                                original_video,
                                "-t",
                                str(duration),
                                "-c:v",
                                "libx264",
                                "-preset",
                                "ultrafast",
                                "-crf",
                                "27",
                                "-an",
                                normal_file
                            ]


                            subprocess.run(
                                normal_cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )


                            if os.path.exists(
                                normal_file
                            ):

                                freeze_parts.append(
                                    normal_file
                                )


                            # -----------------------------------------
                            # Freeze frame
                            # -----------------------------------------

                            if segment_end < video_duration:

                                freeze_file = os.path.join(
                                    temp_dir,
                                    f"freeze_{part_index}.mp4"
                                )


                                freeze_timestamp = (
                                    segment_end
                                    - 0.05
                                )


                                zoom = (
                                    zoom_level
                                    if zoom_enabled
                                    else 1.0
                                )


                                freeze_frames = max(
                                    1,
                                    int(
                                        fps
                                        * freeze_duration
                                    )
                                )


                                # Zoom in → zoom out
                                if zoom_enabled:

                                    zoom_expr = (
                                        f"if("
                                        f"lte(on,"
                                        f"{freeze_frames / 2}),"
                                        f"1+({zoom}-1)*on/"
                                        f"({freeze_frames / 2}),"
                                        f"{zoom}-({zoom}-1)*("
                                        f"on-{freeze_frames / 2}"
                                        f")/"
                                        f"({freeze_frames / 2})"
                                        f")"
                                    )

                                else:

                                    zoom_expr = "1"


                                freeze_cmd = [
                                    "ffmpeg",
                                    "-y",
                                    "-ss",
                                    str(freeze_timestamp),
                                    "-i",
                                    original_video,
                                    "-frames:v",
                                    "1",
                                    "-vf",
                                    (
                                        f"scale="
                                        f"iw*{zoom}:"
                                        f"ih*{zoom},"
                                        f"crop=iw/{zoom}:"
                                        f"ih/{zoom},"
                                        f"zoompan="
                                        f"z='{zoom_expr}':"
                                        f"d={freeze_frames}:"
                                        f"s={width}x{height}:"
                                        f"fps={int(fps)}"
                                    ),
                                    "-t",
                                    str(freeze_duration),
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "ultrafast",
                                    "-crf",
                                    "27",
                                    freeze_file
                                ]


                                subprocess.run(
                                    freeze_cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )


                                if os.path.exists(
                                    freeze_file
                                ):

                                    freeze_parts.append(
                                        freeze_file
                                    )


                            current = segment_end
                            part_index += 1


                        # ---------------------------------------------
                        # Concat freeze parts
                        # ---------------------------------------------

                        concat_file = os.path.join(
                            temp_dir,
                            "freeze_concat.txt"
                        )


                        with open(
                            concat_file,
                            "w",
                            encoding="utf-8"
                        ) as f:

                            for part in freeze_parts:

                                f.write(
                                    "file '"
                                    + part.replace(
                                        "'",
                                        "'\\''"
                                    )
                                    + "'\n"
                                )


                        concat_cmd = [
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
                            processed_video
                        ]


                        subprocess.run(
                            concat_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )


                    else:

                        shutil.copy(
                            original_video,
                            processed_video
                        )


                    # =================================================
                    # FINAL VIDEO WITH SUBTITLE + VOICE
                    # =================================================

                    final_output = os.path.join(
                        temp_dir,
                        "final_movie_recap.mp4"
                    )


                    subtitle_filter_path = (
                        ass_path
                        .replace(
                            "\\",
                            "/"
                        )
                        .replace(
                            ":",
                            "\\:"
                        )
                    )


                    video_filter = (
                        f"ass='{subtitle_filter_path}'"
                    )


                    # -------------------------------------------------
                    # Calculate final video duration
                    # -------------------------------------------------

                    freeze_count = 0

                    if freeze_enabled:

                        freeze_count = int(
                            math.floor(
                                video_duration
                                / freeze_interval
                            )
                        )


                    base_video_duration = (
                        video_duration
                        + freeze_count
                        * freeze_duration
                    )


                    extra_audio_duration = max(
                        0,
                        voice_duration
                        - base_video_duration
                    )


                    # -------------------------------------------------
                    # If voice is longer than video,
                    # freeze last frame.
                    # -------------------------------------------------

                    if extra_audio_duration > 0:

                        video_filter = (
                            f"tpad="
                            f"stop_mode=clone:"
                            f"stop_duration="
                            f"{extra_audio_duration},"
                            f"{video_filter}"
                        )


                    final_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        processed_video,
                        "-i",
                        st.session_state[
                            "voiceover_path"
                        ],
                        "-vf",
                        video_filter,
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-crf",
                        "27",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        "-shortest",
                        "-t",
                        str(
                            max(
                                voice_duration,
                                base_video_duration
                            )
                        ),
                        final_output
                    ]


                    result = subprocess.run(
                        final_cmd,
                        capture_output=True,
                        text=True
                    )


                    if result.returncode != 0:

                        st.error(
                            "❌ FFmpeg Error"
                        )

                        st.code(
                            result.stderr
                        )

                    elif os.path.exists(
                        final_output
                    ):

                        st.success(
                            "🎉 Final video export completed!"
                        )


                        st.video(
                            final_output
                        )


                        with open(
                            final_output,
                            "rb"
                        ) as f:

                            st.download_button(
                                "⬇️ Download Final Movie Recap",
                                f,
                                file_name=(
                                    "final_movie_recap.mp4"
                                ),
                                mime="video/mp4",
                                use_container_width=True
                            )


                except Exception as e:

                    st.error(
                        f"❌ Final Export Error: {e}"
                    )
