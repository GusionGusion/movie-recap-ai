import streamlit as st
import os
from google import genai
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

def get_audio_duration(file_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            capture_output=True,
            text=True
        )

        return float(result.stdout.strip())

    except Exception:
        return 0.0


def ass_time(seconds):
    seconds = max(0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def split_myanmar_text(text, max_chars=65):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []

    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)

        if cut <= 0:
            cut = max_chars

        chunks.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        chunks.append(text)

    return chunks


def wrap_myanmar(text, max_chars=30):
    """
    Maximum 2 lines.
    Short text = 1 line.
    Long text = 2 lines.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    cut = text.rfind(" ", 0, max_chars)

    if cut <= 0:
        cut = max_chars

    line1 = text[:cut].strip()
    remaining = text[cut:].strip()

    if len(remaining) <= max_chars:
        return line1 + r"\N" + remaining

    # Keep only 2 lines
    line2 = remaining[:max_chars].strip()

    return line1 + r"\N" + line2


@st.cache_resource
def load_whisper_model(model_name):
    import whisper
    return whisper.load_model(model_name)


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "🎥 Upload Movie / Video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Maximum 200MB per file"
)


if uploaded_file is not None:

    if (
        "uploaded_file" not in st.session_state
        or st.session_state.get("uploaded_filename") != uploaded_file.name
    ):

        st.session_state["uploaded_file"] = uploaded_file.getvalue()
        st.session_state["uploaded_filename"] = uploaded_file.name

        suffix = os.path.splitext(uploaded_file.name)[1]

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        temp_video.write(uploaded_file.getvalue())
        temp_video.close()

        st.session_state["video_path"] = temp_video.name


video_path = st.session_state.get("video_path")


# =========================================================
# VIDEO INFORMATION
# =========================================================

if video_path and os.path.exists(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps > 0:
        video_duration = frame_count / fps
    else:
        video_duration = 0

    st.success("✅ Video uploaded successfully")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write(f"🎬 **Duration:** {video_duration:.1f} sec")

    with col2:
        st.write(f"📐 **Resolution:** {width}×{height}")

    with col3:
        st.write(f"🎞️ **FPS:** {fps:.0f}")

    with col4:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        st.write(f"💾 **Size:** {size_mb:.2f} MB")

    st.video(video_path)


# =========================================================
# RECAP SETTINGS
# =========================================================

st.subheader("⚙️ Recap Settings")

settings_col1, settings_col2 = st.columns(2)


with settings_col1:

    whisper_model = st.selectbox(
        "🧠 Whisper Model",
        ["tiny", "base"],
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
        [1.0, 1.1, 1.2],
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
        value=8.0,
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

    zoom_level = st.selectbox(
        "🔍 Zoom Level",
        [1.1, 1.2, 1.3, 1.4, 1.5],
        index=1,
        format_func=lambda x: f"{x:.1f}×",
        key="main_zoom_level"
    )


st.caption(
    "⚙️ Set your settings first, then press ONE CLICK to run the full pipeline."
)


# =========================================================
# ONE CLICK BUTTON
# =========================================================

run_all = False

if uploaded_file is not None:

    st.markdown("### 🎬 Ready to Generate")

    run_all = st.button(
        "🎬 ONE CLICK — GENERATE MOVIE RECAP",
        type="primary",
        use_container_width=True
    )


# =========================================================
# 1. TRANSCRIPT
# =========================================================

if video_path and (st.button("1️⃣ Generate Transcript") or run_all):

    with st.spinner("🧠 Transcribing movie..."):

        try:

            model = load_whisper_model(whisper_model)

            transcript_result = model.transcribe(
                video_path,
                language="en"
            )

            transcript = transcript_result.get(
                "text",
                ""
            )

            st.session_state["transcript_result"] = transcript_result
            st.session_state["transcript"] = transcript

            st.success("✅ Transcript completed")

        except Exception as e:

            st.error(f"❌ Transcript error: {e}")


# =========================================================
# SHOW TRANSCRIPT
# =========================================================

if "transcript" in st.session_state:

    st.subheader("📝 Transcript")

    st.text_area(
        "Movie Transcript",
        st.session_state["transcript"],
        height=250
    )


# =========================================================
# 2. SCENE ANALYSIS
# =========================================================

if "transcript_result" in st.session_state:

    if st.button("2️⃣ Scene Analysis") or run_all:

        with st.spinner("🎬 Analyzing scenes..."):

            segments = st.session_state[
                "transcript_result"
            ].get("segments", [])

            scene_data = []

            for i, seg in enumerate(segments):

                start = float(seg.get("start", 0))
                end = float(seg.get("end", start))
                text = seg.get("text", "").strip()

                scene_data.append({
                    "scene": i + 1,
                    "start": start,
                    "end": end,
                    "text": text
                })

            st.session_state["scene_data"] = scene_data

            st.success(
                f"✅ Scene analysis completed: {len(scene_data)} scenes"
            )


# =========================================================
# SHOW SCENE ANALYSIS
# =========================================================

if "scene_data" in st.session_state:

    st.subheader("🎬 Scene Analysis")

    for scene in st.session_state["scene_data"]:

        st.markdown(
            f"**Scene {scene['scene']}** "
            f"({scene['start']:.1f}s → {scene['end']:.1f}s)"
        )

        st.write(scene["text"])


# =========================================================
# 3. AI SCENE ANALYSIS
# =========================================================

if "scene_data" in st.session_state:

    if st.button("3️⃣ AI Scene Analysis") or run_all:

        with st.spinner("🤖 AI analyzing scenes..."):

            try:

                client = genai.Client(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )

                scene_text = json.dumps(
                    st.session_state["scene_data"],
                    ensure_ascii=False,
                    indent=2
                )

                prompt = f"""
You are a professional movie scene analyst.

Analyze the following movie transcript scenes.

IMPORTANT RULES:
1. Use ONLY information present in the transcript.
2. Do not invent events.
3. Keep chronological order.
4. Identify important characters.
5. Identify location when available.
6. Describe important actions.
7. Describe emotions when supported by the transcript.
8. Give a short scene summary.

Return the analysis in numbered scenes.

Transcript scenes:
{scene_text}
"""

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                ai_scene_analysis = response.text

                st.session_state[
                    "ai_scene_analysis"
                ] = ai_scene_analysis

                st.success("✅ AI Scene Analysis completed")

            except Exception as e:

                st.error(
                    f"❌ AI Scene Analysis error: {e}"
                )


# =========================================================
# SHOW AI SCENE ANALYSIS
# =========================================================

if "ai_scene_analysis" in st.session_state:

    st.subheader("🤖 AI Scene Analysis")

    st.text_area(
        "AI Scene Analysis",
        st.session_state["ai_scene_analysis"],
        height=400
    )


# =========================================================
# 4. MOVIE RECAP SCRIPT
# =========================================================

if "ai_scene_analysis" in st.session_state:

    if st.button("4️⃣ Generate Movie Recap Script") or run_all:

        with st.spinner("✍️ Creating recap script..."):

            try:

                client = genai.Client(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )

                prompt = f"""
You are a professional movie recap writer.

Create an engaging movie recap narration from the scene analysis below.

RULES:
1. Use only the information provided.
2. Do not invent events.
3. Do not change the story.
4. Keep chronological order.
5. Make it suitable for voiceover.
6. Divide the narration into numbered scenes.
7. Keep the narration natural and engaging.

AI Scene Analysis:
{st.session_state["ai_scene_analysis"]}
"""

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                recap_script = response.text

                st.session_state[
                    "recap_script"
                ] = recap_script

                st.success("✅ Recap script completed")

            except Exception as e:

                st.error(
                    f"❌ Recap Script error: {e}"
                )


# =========================================================
# SHOW RECAP SCRIPT
# =========================================================

if "recap_script" in st.session_state:

    st.subheader("🎞️ Movie Recap Script")

    st.text_area(
        "Recap Script",
        st.session_state["recap_script"],
        height=400
    )


# =========================================================
# 5. MYANMAR RECAP
# =========================================================

if "recap_script" in st.session_state:

    if st.button("5️⃣ Generate Myanmar Recap") or run_all:

        with st.spinner("🇲🇲 Creating Myanmar narration..."):

            try:

                client = genai.Client(
                    api_key=st.secrets["GEMINI_API_KEY"]
                )

                prompt = f"""
Translate the following movie recap into natural spoken Myanmar.

IMPORTANT:
1. Preserve the exact meaning.
2. Do not add events.
3. Do not remove events.
4. Keep chronological order.
5. Make it suitable for Myanmar voiceover.
6. Use natural spoken Myanmar.
7. Do not include English.
8. Output ONLY the Myanmar narration.
9. End sentences naturally.
10. Replace Burmese sentence ending "တယ်" with "ဒယ်".

Movie Recap:
{st.session_state["recap_script"]}
"""

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                myanmar_recap = response.text

                st.session_state[
                    "myanmar_recap"
                ] = myanmar_recap

                st.success(
                    "✅ Myanmar Recap completed"
                )

            except Exception as e:

                st.error(
                    f"❌ Myanmar Recap error: {e}"
                )


# =========================================================
# SHOW MYANMAR RECAP
# =========================================================

if "myanmar_recap" in st.session_state:

    st.subheader("🇲🇲 Myanmar Recap")

    st.text_area(
        "Myanmar Narration",
        st.session_state["myanmar_recap"],
        height=400
    )


# =========================================================
# 6. MYANMAR VOICEOVER
# =========================================================

if "myanmar_recap" in st.session_state:

    if st.button("6️⃣ Generate Myanmar Voiceover") or run_all:

        with st.spinner("🎙️ Generating Myanmar voiceover..."):

            try:

                import edge_tts

                if "Female" in voice_gender:

                    selected_voice = (
                        "my-MM-NilarNeural"
                    )

                else:

                    selected_voice = (
                        "my-MM-ThihaNeural"
                    )

                if voice_speed == 1.0:
                    tts_rate = "+0%"

                elif voice_speed == 1.1:
                    tts_rate = "+10%"

                else:
                    tts_rate = "+20%"

                chunks = split_myanmar_text(
                    st.session_state["myanmar_recap"],
                    65
                )

                voice_chunks = []

                for i, chunk in enumerate(chunks):

                    chunk_file = os.path.join(
                        tempfile.gettempdir(),
                        f"myanmar_tts_{i}.mp3"
                    )

                    communicate = edge_tts.Communicate(
                        chunk,
                        selected_voice,
                        rate=tts_rate
                    )

                    asyncio.run(
                        communicate.save(chunk_file)
                    )

                    duration = get_audio_duration(
                        chunk_file
                    )

                    voice_chunks.append({
                        "file": chunk_file,
                        "text": chunk,
                        "duration": duration
                    })

                voice_file = os.path.join(
                    tempfile.gettempdir(),
                    "myanmar_voiceover.mp3"
                )

                # =================================================
                # SMOOTH VOICE COMBINATION
                # =================================================

                # IMPORTANT:
                # Do NOT add hard 0.4 sec silence.
                # A short 80ms crossfade makes the speech smoother.

                combined_voice = os.path.join(
                    tempfile.gettempdir(),
                    "combined_myanmar_voice.mp3"
                )

                crossfade = 0.08

                if len(voice_chunks) == 1:

                    shutil.copy2(
                        voice_chunks[0]["file"],
                        combined_voice
                    )

                else:

                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y"
                    ]

                    for chunk in voice_chunks:

                        ffmpeg_cmd += [
                            "-i",
                            chunk["file"]
                        ]

                    filter_parts = []

                    # Normalize all audio chunks

                    for i in range(
                        len(voice_chunks)
                    ):

                        filter_parts.append(
                            f"[{i}:a]"
                            f"aresample=44100,"
                            f"aformat="
                            f"sample_fmts=fltp:"
                            f"sample_rates=44100:"
                            f"channel_layouts=stereo"
                            f"[a{i}]"
                        )

                    current = "a0"

                    for i in range(
                        1,
                        len(voice_chunks)
                    ):

                        output = f"xf{i}"

                        filter_parts.append(
                            f"[{current}][a{i}]"
                            f"acrossfade="
                            f"d={crossfade}:"
                            f"c1=tri:c2=tri"
                            f"[{output}]"
                        )

                        current = output

                    filter_complex = ";".join(
                        filter_parts
                    )

                    ffmpeg_cmd += [
                        "-filter_complex",
                        filter_complex,
                        "-map",
                        f"[{current}]",
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        "128k",
                        combined_voice
                    ]

                    subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                shutil.copy2(
                    combined_voice,
                    voice_file
                )

                voice_duration = get_audio_duration(
                    voice_file
                )

                # =================================================
                # SUBTITLE TIMING
                # =================================================

                subtitle_data = []

                cursor = 0.0

                for i, chunk in enumerate(
                    voice_chunks
                ):

                    duration = float(
                        chunk["duration"]
                    )

                    start_time = cursor
                    end_time = (
                        cursor + duration
                    )

                    subtitle_data.append({
                        "start": start_time,
                        "end": end_time,
                        "text": chunk["text"]
                    })

                    if i < len(
                        voice_chunks
                    ) - 1:

                        cursor = (
                            end_time
                            - crossfade
                        )

                    else:

                        cursor = end_time

                predicted_duration = cursor

                if (
                    predicted_duration > 0
                    and voice_duration > 0
                ):

                    timing_scale = (
                        voice_duration
                        / predicted_duration
                    )

                    timing_scale = max(
                        0.98,
                        min(1.02, timing_scale)
                    )

                else:

                    timing_scale = 1.0

                for item in subtitle_data:

                    item["start"] *= (
                        timing_scale
                    )

                    item["end"] *= (
                        timing_scale
                    )

                st.session_state[
                    "voiceover_file"
                ] = voice_file

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
                ] = voice_chunks

                st.session_state[
                    "voice_speed_value"
                ] = voice_speed

                st.session_state[
                    "selected_voice"
                ] = selected_voice

                st.session_state[
                    "tts_rate"
                ] = tts_rate

                st.success(
                    "✅ Myanmar Voiceover completed"
                )

                st.write(
                    f"🎙️ Voiceover Duration: "
                    f"{voice_duration:.1f} seconds"
                )

                st.audio(
                    voice_file
                )

            except Exception as e:

                st.error(
                    f"❌ Voiceover error: {e}"
                )


# =========================================================
# 7. MYANMAR SUBTITLE
# =========================================================

if "subtitle_data" in st.session_state:

    st.subheader("🇲🇲 Myanmar Subtitle")

    st.caption(
        "Subtitle timing is based on the actual Myanmar TTS segments."
    )

    for i, item in enumerate(
        st.session_state["subtitle_data"]
    ):

        st.write(
            f"**{i + 1}.** "
            f"{item['start']:.2f}s → "
            f"{item['end']:.2f}s"
        )

        st.write(
            item["text"]
        )


# =========================================================
# 8. SUBTITLE EXPORT
# =========================================================

if "subtitle_data" in st.session_state:

    st.subheader("📄 Subtitle Export")

    srt_lines = []

    for i, item in enumerate(
        st.session_state["subtitle_data"],
        start=1
    ):

        start = item["start"]
        end = item["end"]

        def srt_time(seconds):

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

            milliseconds = int(
                (seconds - int(seconds))
                * 1000
            )

            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{secs:02d},"
                f"{milliseconds:03d}"
            )

        srt_lines.append(
            str(i)
        )

        srt_lines.append(
            f"{srt_time(start)} --> "
            f"{srt_time(end)}"
        )

        srt_lines.append(
            item["text"]
        )

        srt_lines.append("")

    srt_content = "\n".join(
        srt_lines
    )

    st.download_button(
        "⬇️ Download Myanmar SRT",
        srt_content,
        file_name="myanmar_subtitles.srt",
        mime="application/x-subrip",
        use_container_width=True
    )


# =========================================================
# 9. SCENE TIMING
# =========================================================

if "subtitle_data" in st.session_state:

    st.subheader("⏱️ Scene Timing")

    voice_duration = st.session_state.get(
        "voice_duration",
        0
    )

    subtitle_data = (
        st.session_state["subtitle_data"]
    )

    fixed_count = 0

    for item in subtitle_data:

        old_end = item["end"]

        item["start"] = max(
            0,
            min(
                item["start"],
                voice_duration
            )
        )

        item["end"] = max(
            item["start"],
            min(
                item["end"],
                voice_duration
            )
        )

        if item["end"] != old_end:
            fixed_count += 1

    st.session_state[
        "subtitle_data"
    ] = subtitle_data

    st.success(
        f"✅ Timing fixed: "
        f"{len(subtitle_data)} subtitles"
    )


# =========================================================
# 10. FREEZE FRAME + ZOOM
# =========================================================

st.subheader("🧊 Freeze Frame + Zoom")

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

    st.write(
        f"🔍 Zoom Level: "
        f"{zoom_level:.1f}×"
    )

    st.caption(
        "Every selected interval → "
        "Freeze + Zoom In → Zoom Out"
    )


# =========================================================
# 11. FINAL VIDEO EXPORT
# =========================================================

if (
    video_path
    and "voiceover_file" in st.session_state
    and "subtitle_data" in st.session_state
):

    if st.button(
        "🎬 FINAL VIDEO EXPORT",
        type="primary",
        use_container_width=True
    ) or run_all:

        with st.spinner(
            "🎬 Rendering final movie recap..."
        ):

            try:

                voice_file = st.session_state[
                    "voiceover_file"
                ]

                voice_duration = get_audio_duration(
                    voice_file
                )

                original_duration = (
                    video_duration
                )

                output_file = os.path.join(
                    tempfile.gettempdir(),
                    "final_movie_recap.mp4"
                )

                # =================================================
                # ASS SUBTITLE
                # =================================================

                ass_file = os.path.join(
                    tempfile.gettempdir(),
                    "myanmar_subtitles.ass"
                )

                with open(
                    ass_file,
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
                        "PlayResY: 1024\n\n"
                    )

                    f.write(
                        "[V4+ Styles]\n"
                    )

                    f.write(
                        "Format: Name, Fontname, "
                        "Fontsize, PrimaryColour, "
                        "SecondaryColour, "
                        "OutlineColour, BackColour, "
                        "Bold, Italic, Underline, "
                        "StrikeOut, ScaleX, ScaleY, "
                        "Spacing, Angle, BorderStyle, "
                        "Outline, Shadow, Alignment, "
                        "MarginL, MarginR, MarginV, Encoding\n"
                    )

                    # Yellow text + black outline
                    # Font size = 56
                    f.write(
                        "Style: Myanmar,"
                        "Noto Sans Myanmar,"
                        "56,"
                        "&H0000FFFF,"
                        "&H0000FFFF,"
                        "&H00000000,"
                        "&H00000000,"
                        "0,0,0,0,"
                        "100,100,0,0,"
                        "1,3,1,2,"
                        "30,30,45,1\n\n"
                    )

                    f.write(
                        "[Events]\n"
                    )

                    f.write(
                        "Format: Layer, Start, End, "
                        "Style, Name, MarginL, "
                        "MarginR, MarginV, Effect, Text\n"
                    )

                    for item in st.session_state[
                        "subtitle_data"
                    ]:

                        start = max(
                            0,
                            item["start"]
                        )

                        end = min(
                            voice_duration,
                            item["end"]
                        )

                        if end <= start:
                            continue

                        text = wrap_myanmar(
                            item["text"],
                            30
                        )

                        f.write(
                            f"Dialogue: 0,"
                            f"{ass_time(start)},"
                            f"{ass_time(end)},"
                            f"Myanmar,"
                            f",0,0,0,,"
                            f"{text}\n"
                        )

                # =================================================
                # VIDEO FILTER
                # =================================================

                if freeze_enabled:

                    interval = float(
                        freeze_interval
                    )

                    freeze_time = float(
                        freeze_duration
                    )

                    zoom_amount = float(
                        zoom_level
                    )

                    parts = []

                    current = 0.0
                    freeze_index = 0

                    while current < original_duration:

                        normal_end = min(
                            current + interval,
                            original_duration
                        )

                        segment_duration = (
                            normal_end - current
                        )

                        if segment_duration > 0:

                            parts.append(
                                (
                                    f"[0:v]"
                                    f"trim="
                                    f"start={current}:"
                                    f"end={normal_end},"
                                    f"setpts=PTS-STARTPTS,"
                                    f"scale=576:1024:"
                                    f"force_original_aspect_ratio=decrease,"
                                    f"pad=576:1024:"
                                    f"(ow-iw)/2:"
                                    f"(oh-ih)/2"
                                    f"[v{freeze_index}]"
                                )
                            )

                            freeze_index += 1

                        if normal_end >= (
                            original_duration
                        ):
                            break

                        # Freeze frame
                        freeze_start = max(
                            current,
                            normal_end - 0.10
                        )

                        parts.append(
                            (
                                f"[0:v]"
                                f"trim="
                                f"start={freeze_start}:"
                                f"end={normal_end},"
                                f"setpts=PTS-STARTPTS,"
                                f"select='eq(n,0)',"
                                f"loop="
                                f"loop=-1:"
                                f"size=1:"
                                f"start=0,"
                                f"trim="
                                f"duration={freeze_time},"
                                f"setpts=PTS-STARTPTS,"
                                f"scale=576:1024:"
                                f"force_original_aspect_ratio=decrease,"
                                f"pad=576:1024:"
                                f"(ow-iw)/2:"
                                f"(oh-ih)/2,"
                                f"zoompan="
                                f"z='if("
                                f"lte(on,29),"
                                f"1+({zoom_amount}-1)*on/29,"
                                f"{zoom_amount}-"
                                f"({zoom_amount}-1)*"
                                f"(on-29)/29"
                                f")':"
                                f"d=60:"
                                f"s=576x1024:"
                                f"fps=30"
                                f"[v{freeze_index}]"
                            )
                        )

                        freeze_index += 1

                        current = normal_end

                    video_labels = [
                        f"[v{i}]"
                        for i in range(
                            freeze_index
                        )
                    ]

                    concat_inputs = "".join(
                        video_labels
                    )

                    parts.append(
                        f"{concat_inputs}"
                        f"concat=n={freeze_index}:"
                        f"v=1:a=0,"
                        f"setpts=PTS-STARTPTS"
                        f"[vbase]"
                    )

                    video_filter = ";".join(
                        parts
                    )

                else:

                    video_filter = (
                        "[0:v]"
                        "scale=576:1024:"
                        "force_original_aspect_ratio=decrease,"
                        "pad=576:1024:"
                        "(ow-iw)/2:"
                        "(oh-ih)/2"
                        "[vbase]"
                    )

                # =================================================
                # FINAL FFMPEG
                # =================================================

                if voice_duration > original_duration:

                    extra_duration = (
                        voice_duration
                        - original_duration
                    )

                    final_video_filter = (
                        video_filter
                        + ";"
                        + f"[vbase]"
                        f"tpad="
                        f"stop_mode=clone:"
                        f"stop_duration={extra_duration},"
                        f"setpts=PTS-STARTPTS"
                        f"[vfinal]"
                    )

                else:

                    final_video_filter = (
                        video_filter
                        + ";"
                        "[vbase]"
                        "setpts=PTS-STARTPTS"
                        "[vfinal]"
                    )

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",

                    "-i",
                    video_path,

                    "-i",
                    voice_file,

                    "-filter_complex",
                    final_video_filter,

                    "-i",
                    video_path,

                    "-filter_complex",
                    final_video_filter
                ]

                # Use a simpler final command
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",

                    "-i",
                    video_path,

                    "-i",
                    voice_file,

                    "-filter_complex",
                    (
                        final_video_filter
                        + ";"
                        "[vfinal]"
                        f"subtitles="
                        f"'{ass_file}'"
                        "[vout]"
                    ),

                    "-map",
                    "[vout]",

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

                    "-t",
                    str(voice_duration),

                    "-movflags",
                    "+faststart",

                    output_file
                ]

                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:

                    st.error(
                        "❌ FFmpeg Export Error"
                    )

                    st.code(
                        result.stderr[-5000:]
                    )

                else:

                    st.session_state[
                        "final_video"
                    ] = output_file

                    st.success(
                        "🎉 Final Video Export Completed!"
                    )

                    st.video(
                        output_file
                    )

                    with open(
                        output_file,
                        "rb"
                    ) as video_bytes:

                        st.download_button(
                            "⬇️ Download Final Movie Recap",
                            video_bytes,
                            file_name=(
                                "final_movie_recap.mp4"
                            ),
                            mime="video/mp4",
                            use_container_width=True
                        )

            except Exception as e:

                st.error(
                    f"❌ Final Export error: {e}"
                )
