import streamlit as st
import os
from google import genai
import cv2
import tempfile
import subprocess
import math
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
# VIDEO UPLOAD
# =========================================================

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


    st.session_state["video_path"] = video_path

    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():

        st.error(
            "❌ Video file could not be opened."
        )

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

        st.divider()

        st.subheader(
            "🎥 Video Preview"
        )

        st.video(
            uploaded_file
        )

    cap.release()

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
    ):

        st.info(
            "⏳ Transcribing movie audio... Please wait."
        )

        try:

            import whisper

            model = whisper.load_model(
                "base"
            )

            result = model.transcribe(
                video_path,
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
        st.session_state["transcript"],
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

                    st.write(
                        text
                    )


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
        ]
    )


st.divider()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_audio_duration(audio_file):

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_file
        ],
        capture_output=True,
        text=True,
        check=True
    )

    return float(
        result.stdout.strip()
    )


def split_myanmar_text(
    text,
    max_chars=65
):
    """
    Split Myanmar narration into subtitle/TTS chunks.

    We prefer sentence boundaries first.
    If a sentence is too long, split it into
    smaller pieces.
    """

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    if not text:
        return []


    # -----------------------------------------------------
    # First split by Myanmar / normal sentence punctuation
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Break very long sentences
    # -----------------------------------------------------

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
    """
    Wrap subtitle text into maximum 3 lines.
    """

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

                lines.append(
                    current
                )

            current = word


    if current:

        lines.append(
            current
        )


    # Maximum 3 lines

    if len(lines) > 3:

        lines = lines[:3]


    return "\\N".join(
        lines
    )


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


# =========================================================
# MYANMAR FEMALE VOICEOVER
#
# IMPORTANT:
# Voiceover and subtitle timing are generated together.
#
# NO WHISPER is used for subtitle timing.
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
        "Subtitle timing will be created directly "
        "from the actual generated voice segments."
    )


    if st.button(
        "🎙️ Generate Myanmar Voiceover"
    ):

        try:

            import edge_tts
            import asyncio


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


            # =========================================
            # Split narration
            # =========================================

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


            # =========================================
            # Temporary working folder
            # =========================================

            work_dir = tempfile.mkdtemp(
                prefix="movie_recap_voice_"
            )


            raw_files = []
            wav_files = []

            subtitle_data = []


            # =========================================
            # Generate each voice segment
            # =========================================

            async def create_segment_voice(
                segment_text,
                output_file
            ):

                communicate = edge_tts.Communicate(
                    segment_text,
                    "my-MM-NilarNeural"
                )

                await communicate.save(
                    output_file
                )


            cumulative_time = 0.0


            progress = st.progress(
                0
            )


            for index, chunk in enumerate(
                chunks
            ):

                raw_file = os.path.join(
                    work_dir,
                    f"raw_{index:04d}.mp3"
                )

                wav_file = os.path.join(
                    work_dir,
                    f"segment_{index:04d}.wav"
                )


                # -------------------------------------
                # TTS
                # -------------------------------------

                asyncio.run(
                    create_segment_voice(
                        chunk,
                        raw_file
                    )
                )


                raw_files.append(
                    raw_file
                )


                # -------------------------------------
                # Apply voice speed AND convert to WAV
                #
                # WAV duration is measured AFTER speed.
                # Therefore subtitle timing exactly follows
                # the final voice speed.
                # -------------------------------------

                ffmpeg_args = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    raw_file
                ]


                if voice_speed != 1.0:

                    ffmpeg_args += [
                        "-filter:a",
                        f"atempo={voice_speed}"
                    ]


                ffmpeg_args += [
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    wav_file
                ]


                subprocess.run(
                    ffmpeg_args,
                    capture_output=True,
                    text=True,
                    check=True
                )


                wav_files.append(
                    wav_file
                )


                # -------------------------------------
                # Measure EXACT resulting segment duration
                # -------------------------------------

                segment_duration = (
                    get_audio_duration(
                        wav_file
                    )
                )


                start_time = (
                    cumulative_time
                )

                end_time = (
                    cumulative_time
                    + segment_duration
                )


                subtitle_data.append({
                    "start": start_time,
                    "end": end_time,
                    "text": chunk
                })


                cumulative_time = end_time


                progress.progress(
                    (index + 1) / len(chunks)
                )


            # =========================================
            # Create concat list
            # =========================================

            concat_file = os.path.join(
                work_dir,
                "concat.txt"
            )


            with open(
                concat_file,
                "w",
                encoding="utf-8"
            ) as f:

                for wav_file in wav_files:

                    absolute_path = os.path.abspath(
                        wav_file
                    ).replace(
                        "'",
                        "'\\''"
                    )

                    f.write(
                        f"file '{absolute_path}'\n"
                    )


            # =========================================
            # Combine all WAV segments
            # =========================================

            combined_wav = os.path.join(
                work_dir,
                "combined.wav"
            )


            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_file,
                    "-c:a",
                    "pcm_s16le",
                    combined_wav
                ],
                capture_output=True,
                text=True,
                check=True
            )


            # =========================================
            # Convert final audio to MP3
            # =========================================

            final_voice_file = os.path.join(
                work_dir,
                "myanmar_voiceover.mp3"
            )


            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    combined_wav,
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    final_voice_file
                ],
                capture_output=True,
                text=True,
                check=True
            )


            # =========================================
            # Get final exact duration
            # =========================================

            voice_duration = (
                get_audio_duration(
                    final_voice_file
                )
            )


            # =========================================
            # Small final correction
            #
            # Keep subtitle timing inside final audio.
            # =========================================

            corrected_subtitles = []


            for item in subtitle_data:

                start = float(
                    item["start"]
                )

                end = float(
                    item["end"]
                )

                if start >= voice_duration:

                    continue


                end = min(
                    end,
                    voice_duration
                )


                if end <= start:

                    continue


                corrected_subtitles.append({
                    "start": start,
                    "end": end,
                    "text": item["text"]
                })


            # =========================================
            # Save session state
            # =========================================

            st.session_state[
                "voiceover_file"
            ] = final_voice_file


            st.session_state[
                "voice_duration"
            ] = voice_duration


            st.session_state[
                "subtitle_data"
            ] = corrected_subtitles


            st.session_state[
                "subtitle_timing_source"
            ] = "tts_segments"


            st.session_state[
                "voice_chunks"
            ] = chunks


            st.success(
                "✅ Myanmar Female Voiceover generated!"
            )


            st.success(
                f"✅ {len(corrected_subtitles)} subtitles "
                f"were synced directly to the voice segments."
            )


            st.info(
                f"🎙️ Voiceover Duration: "
                f"{voice_duration:.2f} seconds"
            )


            st.info(
                "🔊 Subtitle timing source: "
                "Actual TTS audio duration"
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
#
# IMPORTANT:
# Subtitle data was already generated together with
# the voiceover.
#
# NO WHISPER
# NO GEMINI TIMESTAMP GUESSING
# =========================================================

st.subheader(
    "🇲🇲 Myanmar Subtitle"
)


if "subtitle_data" in st.session_state:

    st.success(
        "✅ Subtitle timing is synchronized with the generated voiceover."
    )


    st.text_area(
        "💬 Myanmar Subtitle Preview",
        "\n".join(
            [
                f'{item["start"]:.2f}s → '
                f'{item["end"]:.2f}s | '
                f'{item["text"]}'
                for item in st.session_state[
                    "subtitle_data"
                ]
            ]
        ),
        height=400
    )


    st.caption(
        "ℹ️ Subtitle timing is generated from each actual "
        "TTS audio segment. Whisper is not used."
    )


st.divider()


# =========================================================
# SUBTITLE EXPORT
# =========================================================

st.subheader(
    "📥 Subtitle Export"
)


if "subtitle_data" in st.session_state:

    srt_content = ""


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


        srt_content += (
            f"{i}\n"
            f"{start_time} --> "
            f"{end_time}\n"
            f"{text}\n\n"
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

        voice_duration = (
            get_audio_duration(
                st.session_state[
                    "voiceover_file"
                ]
            )
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

        start = float(
            item["start"]
        )

        end = float(
            item["end"]
        )

        text = str(
            item["text"]
        ).strip()


        if not text:
            continue


        start = max(
            0,
            start
        )


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


    st.session_state[
        "subtitle_timing_source"
    ] = "tts_segments"


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
    value=True
)


freeze_interval = st.number_input(
    "⏱️ Freeze Every",
    min_value=5.0,
    max_value=60.0,
    value=10.0,
    step=1.0
)


freeze_duration = st.number_input(
    "🧊 Freeze Duration",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
)


st.caption(
    "Every 10 seconds → "
    "2 seconds Freeze + Zoom In → Zoom Out"
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

            # =========================================
            # Save original video
            # =========================================

            with open(
                "original_movie.mp4",
                "wb"
            ) as f:

                uploaded = (
                    st.session_state[
                        "uploaded_file"
                    ]
                )

                if isinstance(
                    uploaded,
                    bytes
                ):

                    f.write(
                        uploaded
                    )

                else:

                    f.write(
                        uploaded.getbuffer()
                    )


            voice_file = (
                st.session_state[
                    "voiceover_file"
                ]
            )


            # =========================================
            # Original video duration
            # =========================================

            video_duration = get_audio_duration(
                "original_movie.mp4"
            )


            # =========================================
            # Exact voiceover duration
            # =========================================

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


            # =========================================
            # Freeze settings
            # =========================================

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


            # =========================================
            # ASS subtitle header
            # =========================================

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


            # =========================================
            # Add subtitles
            #
            # IMPORTANT:
            # Direct voiceover timeline.
            # NO freeze offset.
            # =========================================

            for item in st.session_state[
                "subtitle_data"
            ]:

                new_start = max(
                    0,
                    float(item["start"])
                )

                new_end = max(
                    new_start + 0.1,
                    float(item["end"])
                )


                text = str(
                    item["text"]
                ).strip()


                text = wrap_myanmar(
                    text,
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


            # =========================================
            # Save ASS
            # =========================================

            with open(
                "myanmar_subtitles.ass",
                "w",
                encoding="utf-8-sig"
            ) as f:

                f.write(
                    ass_content
                )


            # =========================================
            # Build video filters
            # =========================================

            filter_parts = []

            labels = []


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


                    # ---------------------------------
                    # Normal segment
                    # ---------------------------------

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


                    # ---------------------------------
                    # Freeze + Zoom
                    # ---------------------------------

                    if end < video_duration:

                        freeze_label = (
                            f"freeze{i}"
                        )


                        frame_time = max(
                            start,
                            end - 0.10
                        )


                        freeze_filter = (
                            f"[0:v]"
                            f"trim=start={frame_time}:"
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


            # =========================================
            # Subtitle overlay
            # =========================================

            filter_parts.append(
                "[basevideo]"
                "ass=myanmar_subtitles.ass"
                "[vout]"
            )


            # =========================================
            # Actual freeze count
            # =========================================

            if freeze_enabled:

                actual_freeze_count = max(
                    0,
                    segment_count - 1
                )

            else:

                actual_freeze_count = 0


            # =========================================
            # Base video duration
            # =========================================

            base_video_duration = (
                video_duration
                + (
                    actual_freeze_count
                    * freeze_time
                )
            )


            # =========================================
            # Extra duration
            # =========================================

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


            # =========================================
            # Extend final frame if needed
            # =========================================

            if extra_duration > 0:

                filter_parts.append(
                    "[vout]"
                    f"tpad=stop_mode=clone:"
                    f"stop_duration={extra_duration:.3f}:"
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


            # =========================================
            # Filter complex
            # =========================================

            filter_complex = ";".join(
                filter_parts
            )


            # =========================================
            # Output
            # =========================================

            output_video = (
                "final_movie_recap.mp4"
            )


            command = [

                "ffmpeg",

                "-y",

                "-i",
                "original_movie.mp4",

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
                "veryfast",

                "-crf",
                "23",

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


            # =========================================
            # Run FFmpeg
            # =========================================

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

                # =====================================
                # Verify final duration
                # =====================================

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
                        f,
                        file_name=(
                            "final_movie_recap.mp4"
                        ),
                        mime="video/mp4"
                    )


        except Exception as e:

            st.error(
                f"❌ Final Video Export Error: {e}"
            )
