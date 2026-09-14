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
# PAGE SETTINGS & SESSION STATES
# =========================================================
st.set_page_config(
    page_title="Movie Recap AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "processing" not in st.session_state:
    st.session_state["processing"] = False

if "process_percent" not in st.session_state:
    st.session_state["process_percent"] = 0

if "process_step" not in st.session_state:
    st.session_state["process_step"] = ""

if "process_complete" not in st.session_state:
    st.session_state["process_complete"] = False

# =========================================================
# COLOR THEME / CUSTOM UI
# =========================================================
st.markdown(
    """
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background: radial-gradient(
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

    /* MAIN CONTENT */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* TITLE */
    h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        background: linear-gradient(
            90deg,
            #a855f7,
            #6366f1,
            #38bdf8
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.25);
    }

    h2 { color: #e9d5ff !important; font-weight: 750 !important; }
    h3 { color: #c4b5fd !important; font-weight: 700 !important; }
    p, label, .stMarkdown, .stCaption { color: #e5e7eb; }

    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(139, 92, 246, 0.8),
            rgba(59, 130, 246, 0.8),
            transparent
        ) !important;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.90),
            rgba(15, 23, 42, 0.95)
        );
        border: 1px solid rgba(139, 92, 246, 0.45);
        border-radius: 18px;
        padding: 12px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35), 0 0 25px rgba(99, 102, 241, 0.08);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.75),
            rgba(49, 46, 129, 0.30)
        ) !important;
        border: 1px dashed rgba(167, 139, 250, 0.7) !important;
        border-radius: 14px !important;
    }

    /* BUTTONS */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.5);
        background: linear-gradient(
            135deg,
            #6d28d9,
            #4f46e5
        );
        color: white;
        font-weight: 700;
        padding: 0.65rem 1rem;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.25);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #c4b5fd;
        background: linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.40);
    }

    button[kind="primary"] {
        background: linear-gradient(
            90deg,
            #7c3aed,
            #4f46e5,
            #2563eb
        ) !important;
        border: 1px solid rgba(196, 181, 253, 0.65) !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.40) !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(
            90deg,
            #8b5cf6,
            #6366f1,
            #3b82f6
        ) !important;
        transform: translateY(-2px);
    }

    div[data-baseweb="select"] > div, input, textarea {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(139, 92, 246, 0.35) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.92),
            rgba(49, 46, 129, 0.25)
        );
        border: 1px solid rgba(139, 92, 246, 0.30);
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stMetricLabel"] { color: #c4b5fd !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 800 !important; }

    video {
        border-radius: 16px !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.45) !important;
    }

    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 14px;
        margin: 10px 0;
    }
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.08));
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 14px;
        padding: 14px;
        margin: 10px 0;
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
        if not os.path.exists(media_file) or os.path.getsize(media_file) <= 0:
            return 0.0
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", media_file
            ],
            capture_output=True, text=True, check=True
        )
        value = result.stdout.strip()
        return float(value) if value else 0.0
    except Exception:
        return 0.0

def ass_time(seconds):
    seconds = max(0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def split_myanmar_text(text, max_chars=65):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    sentences = re.split(r"(?<=[။!?])\s+|(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue
        words = sentence.split()
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word
        if current:
            chunks.append(current)
    return chunks

def wrap_myanmar(text, max_chars=20):
    text = str(text).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    words = text.split()
    if len(words) > 1:
        best_split = None
        best_score = None
        for i in range(1, len(words)):
            line1 = " ".join(words[:i])
            line2 = " ".join(words[i:])
            longest = max(len(line1), len(line2))
            difference = abs(len(line1) - len(line2))
            score = (longest * 2 + difference)
            if best_score is None or score < best_score:
                best_score = score
                best_split = (line1, line2)
        if best_split:
            return best_split[0] + "\\N" + best_split[1]
    midpoint = math.ceil(len(text) / 2)
    punctuation_positions = [
        text.rfind("၊", 0, midpoint + 5),
        text.rfind(" ", 0, midpoint + 5)
    ]
    valid_positions = [p for p in punctuation_positions if p > 0]
    split_position = max(valid_positions) if valid_positions else midpoint
    line1 = text[:split_position].strip()
    line2 = text[split_position:].strip()
    if line1 and line2:
        return line1 + "\\N" + line2
    return text

def extract_frame_bytes(cap, timestamp):
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, float(timestamp)) * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        max_side = 768
        h, w = frame.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return encoded.tobytes() if success else None
    except Exception:
        return None

@st.cache_resource
def load_whisper_model(model_name="base"):
    import whisper
    return whisper.load_model(model_name)

# =========================================================
# MAIN RECAP PROCESSING ENGINE
# =========================================================
def process_movie_recap():
    try:
        # Step 1: Transcribing (10%)
        st.session_state["process_percent"] = 10
        st.session_state["process_step"] = "🎤 Transcribing movie with Whisper..."
        st.rerun()

    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.session_state["processing"] = False

# ACTUAL EXECUTION IF PROCESSING STATE IS TRUE
if st.session_state["processing"] and not st.session_state["process_complete"]:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center; padding:25px 10px 10px 10px;">
        <div style="font-size:42px; font-weight:800; background:linear-gradient(90deg, #a855f7, #6366f1, #38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🎬 Movie Recap AI
        </div>
        <div style="margin-top:10px; color:#cbd5e1; font-size:18px;">
            Creating your movie recap...
        </div>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(st.session_state["process_percent"] / 100)
    st.write(f"**Status:** {st.session_state['process_step']}")

    # EXECUTE BACKEND PROCESSES IN ORDER
    try:
        # 1. WHISPER TRANSCRIPT
        if st.session_state["process_percent"] < 15:
            model = load_whisper_model(st.session_state["selected_whisper_model"])
            result = model.transcribe(st.session_state["video_path"], language="en")
            st.session_state["transcript_result"] = result
            st.session_state["transcript"] = result["text"]
            st.session_state["process_percent"] = 25
            st.session_state["process_step"] = "🎬 Analyzing scenes with Gemini..."
            st.rerun()

        # 2. GEMINI SCENE ANALYSIS
        elif st.session_state["process_percent"] < 35:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)
            segments = st.session_state["transcript_result"].get("segments", [])
            cap = cv2.VideoCapture(st.session_state["video_path"])
            scene_items = []
            usable_segments = [seg for seg in segments if str(seg.get("text", "")).strip()]
            max_frames = 24
            if len(usable_segments) > max_frames:
                selected_indexes = [round(n * (len(usable_segments) - 1) / (max_frames - 1)) for n in range(max_frames)]
                selected_segments = [usable_segments[i] for i in selected_indexes]
            else:
                selected_segments = usable_segments

            for seg in selected_segments:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", start))
                text = str(seg.get("text", "")).strip()
                timestamp = (start + end) / 2
                frame_bytes = extract_frame_bytes(cap, timestamp)
                if frame_bytes:
                    scene_items.append({"start": start, "end": end, "text": text, "image": frame_bytes})
            cap.release()

            prompt = """You are analyzing an actual movie/video. LOOK AT THE ACTUAL VIDEO FRAME FIRST.
            Then use the transcript only to understand the spoken words. Return ONLY a concise scene analysis."""
            
            contents = [types.Part.from_text(text=prompt)]
            for index, item in enumerate(scene_items, start=1):
                label = f"\n\nSCENE {index}\nTimestamp: {item['start']:.2f}s → {item['end']:.2f}s\nTranscript: {item['text']}\n"
                contents.append(types.Part.from_text(text=label))
                contents.append(types.Part.from_bytes(data=item["image"], mime_type="image/jpeg"))

            response = client.models.generate_content(model="gemini-3.6-flash", contents=contents)
            st.session_state["ai_scene_analysis"] = response.text if response else ""
            st.session_state["process_percent"] = 45
            st.session_state["process_step"] = "📝 Generating Recap Script..."
            st.rerun()

        # 3. GEMINI RECAP SCRIPT
        elif st.session_state["process_percent"] < 50:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)
            scene_analysis = st.session_state["ai_scene_analysis"]
            prompt = f"You are a professional movie recap script writer. Write the recap using ONLY the Scene Analysis below:\n\n{scene_analysis}"
            response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            st.session_state["recap_script"] = response.text
            st.session_state["process_percent"] = 60
            st.session_state["process_step"] = "🇲🇲 Translating to Myanmar..."
            st.rerun()

        # 4. MYANMAR TRANSLATION
        elif st.session_state["process_percent"] < 65:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)
            recap_script = st.session_state["recap_script"]
            prompt = f"Translate the following movie recap narration into natural spoken Myanmar Burmese:\n\n{recap_script}"
            response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            st.session_state["myanmar_recap"] = response.text
            st.session_state["process_percent"] = 75
            st.session_state["process_step"] = "🎙️ Generating Myanmar Voiceover..."
            st.rerun()

        # 5. EDGE-TTS VOICEOVER & SUBTITLE TIMING
        elif st.session_state["process_percent"] < 80:
            import edge_tts
            text = st.session_state["myanmar_recap"].strip()
            chunks = split_myanmar_text(text, max_chars=100)
            work_dir = tempfile.mkdtemp(prefix="movie_recap_voice_")
            
            selected_voice = "my-MM-NilarNeural" if st.session_state["selected_voice_gender"].startswith("👩") else "my-MM-ThihaNeural"
            speed = float(st.session_state["selected_voice_speed"])
            tts_rate = "+0%" if speed == 1.0 else ("+10%" if speed == 1.1 else "+20%")

            raw_files = []
            async def create_all_tts():
                results = []
                for index, chunk in enumerate(chunks):
                    raw_file = os.path.join(work_dir, f"raw_{index:04d}.mp3")
                    communicate = edge_tts.Communicate(text=chunk, voice=selected_voice, rate=tts_rate)
                    await communicate.save(raw_file)
                    results.append(raw_file)
                return results

            raw_files = asyncio.run(create_all_tts())
            TTS_CROSSFADE = 0.08
            normalized_files = []
            raw_durations = []

            for index, raw_file in enumerate(raw_files):
                normalized_file = os.path.join(work_dir, f"normalized_{index:04d}.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-i", raw_file,
                    "-af", "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo",
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", normalized_file
                ], capture_output=True)
                normalized_files.append(normalized_file)
                raw_durations.append(get_audio_duration(normalized_file))

            combined_audio = os.path.join(work_dir, "combined.wav")
            if len(normalized_files) == 1:
                shutil.copyfile(normalized_files[0], combined_audio)
            else:
                ffmpeg_inputs = []
                for file in normalized_files:
                    ffmpeg_inputs.extend(["-i", file])
                filter_parts = []
                previous_label = "[0:a]"
                for i in range(1, len(normalized_files)):
                    output_label = f"[a{i}]"
                    filter_parts.append(f"{previous_label}[{i}:a]acrossfade=d={TTS_CROSSFADE}:c1=tri:c2=tri{output_label}")
                    previous_label = output_label
                filter_complex = ";".join(filter_parts)
                subprocess.run(["ffmpeg", "-y"] + ffmpeg_inputs + ["-filter_complex", filter_complex, "-map", previous_label, "-c:a", "pcm_s16le", combined_audio], capture_output=True)

            final_voice_file = os.path.join(work_dir, "myanmar_voiceover.mp3")
            subprocess.run(["ffmpeg", "-y", "-i", combined_audio, "-c:a", "libmp3lame", "-b:a", "192k", final_voice_file], capture_output=True)
            voice_duration = get_audio_duration(final_voice_file)

            subtitle_data = []
            current_time = 0.0
            for index, chunk in enumerate(chunks):
                raw_duration = raw_durations[index]
                start_time = current_time
                end_time = start_time + raw_duration
                next_time = min(end_time - TTS_CROSSFADE if index < len(chunks) - 1 else end_time, voice_duration)
                if next_time <= start_time:
                    next_time = min(voice_duration, start_time + 0.05)
                if start_time < voice_duration:
                    subtitle_data.append({"start": start_time, "end": next_time, "text": chunk})
                current_time = max(end_time - TTS_CROSSFADE, 0)

            st.session_state["voiceover_file"] = final_voice_file
            st.session_state["voice_duration"] = voice_duration
            st.session_state["subtitle_data"] = subtitle_data
            st.session_state["process_percent"] = 85
            st.session_state["process_step"] = "🧊 Applying Freeze Frame + Zoom & FFmpeg Rendering..."
            st.rerun()

        # 6. FFMPEG FINAL EXPORT
        elif st.session_state["process_percent"] < 100:
            original_video = os.path.join(tempfile.gettempdir(), "movie_recap_original.mp4")
            with open(original_video, "wb") as f:
                f.write(st.session_state["uploaded_file"])
            
            voice_file = st.session_state["voiceover_file"]
            video_duration = get_audio_duration(original_video)
            voice_duration = st.session_state["voice_duration"]
            freeze_enabled = st.session_state["selected_freeze_enabled"]
            interval = float(st.session_state["selected_freeze_interval"]) if freeze_enabled else video_duration + 1
            freeze_time = float(st.session_state["selected_freeze_duration"]) if freeze_enabled else 0

            # Generate ASS Subtitles
            ass_content = """[Script Info]\nScriptType: v4.00+\nPlayResX: 576\nPlayResY: 1024\nScaledBorderAndShadow: yes\n\n[V4+Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Myanmar,Noto Sans Myanmar,68,&H0000FFFF,&H0000FFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,22,22,55,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""
            for item in st.session_state["subtitle_data"]:
                text = wrap_myanmar(item["text"], 16).replace("{", "\\{").replace("}", "\\}")
                ass_content += f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Myanmar,,0,0,0,,{text}\n"

            ass_file = os.path.join(tempfile.gettempdir(), "myanmar_subtitles.ass")
            with open(ass_file, "w", encoding="utf-8-sig") as f:
                f.write(ass_content)

            filter_parts = []
            labels = []
            if freeze_enabled:
                segment_count = int(math.ceil(video_duration / interval))
                for i in range(segment_count):
                    start = i * interval
                    end = min((i + 1) * interval, video_duration)
                    normal_label = f"normal{i}"
                    filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,scale=576:1024,setsar=1[{normal_label}]")
                    labels.append(f"[{normal_label}]")
                    if end < video_duration:
                        freeze_label = f"freeze{i}"
                        frame_time = max(start, end - 0.10)
                        filter_parts.append(f"[0:v]trim=start={frame_time}:end={frame_time+0.0334},setpts=PTS-STARTPTS,select='eq(n,0)',scale=576:1024,setsar=1,zoompan=z='if(lte(on,29),1+0.15*on/29,1.15-0.15*(on-29)/29)':d=60:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=576x1024:fps=30[{freeze_label}]")
                        labels.append(f"[{freeze_label}]")
                concat_inputs = "".join(labels)
                filter_parts.append(f"{concat_inputs}concat=n={len(labels)}:v=1:a=0:unsafe=1,format=yuv420p[basevideo]")
            else:
                filter_parts.append("[0:v]scale=576:1024,setsar=1,format=yuv420p[basevideo]")

            filter_parts.append(f"[basevideo]ass={ass_file}[vout]")

            actual_freeze_count = max(0, int(math.ceil(video_duration / interval)) - 1) if freeze_enabled else 0
            base_video_duration = video_duration + (actual_freeze_count * freeze_time)
            extra_duration = max(0.0, voice_duration - base_video_duration)

            if extra_duration > 0:
                filter_parts.append(f"[vout]tpad=stop_mode=clone:stop_duration={extra_duration:.3f}:start_mode=clone,setpts=PTS-STARTPTS[vout2]")
                final_video_label = "[vout2]"
            else:
                final_video_label = "[vout]"

            filter_complex = ";".join(filter_parts)
            output_video = os.path.join(tempfile.gettempdir(), "final_movie_recap.mp4")
            
            command = [
                "ffmpeg", "-y", "-i", original_video, "-i", voice_file,
                "-filter_complex", filter_complex, "-map", final_video_label, "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "27", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-t", f"{voice_duration:.3f}", output_video
            ]
            
            subprocess.run(command, capture_output=True, text=True)
            st.session_state["final_output_video"] = output_video
            st.session_state["process_percent"] = 100
            st.session_state["process_step"] = "🎉 Complete!"
            st.session_state["process_complete"] = True
            st.rerun()

    except Exception as e:
        st.error(f"❌ Processing Error: {e}")
        st.session_state["processing"] = False

# =========================================================
# UI HEADER & SETTINGS (DISPLAYED WHEN NOT PROCESSING)
# =========================================================
if not st.session_state["processing"]:
    st.title("🎬 Movie Recap AI")
    st.write("Upload a movie and analyze video information.")

    st.markdown(
        """
        <div class="info-box">
        🎥 <b>Video → Transcript → Scene Analysis → Recap Script
        → Myanmar Voiceover → Myanmar Subtitle → Final Video</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # VIDEO UPLOAD
    uploaded_file = st.file_uploader(
        "🎥 Upload Movie / Video",
        type=["mp4", "mov", "avi", "mkv", "webm"]
    )

    if uploaded_file is not None:
        video_bytes = uploaded_file.getvalue()
        st.session_state["uploaded_file"] = video_bytes
        file_size_mb = (uploaded_file.size / (1024 * 1024))
        suffix = os.path.splitext(uploaded_file.name)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            video_path = temp_file.name
        
        st.session_state["video_path"] = video_path
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_seconds = (frame_count / fps) if fps > 0 else 0
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            
            st.markdown('<div class="success-box"> ✅ <b>Video uploaded successfully!</b> </div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Size", f"{file_size_mb:.2f} MB")
                st.metric("Resolution", f"{width} × {height}")
            with col2:
                st.metric("Duration", f"{minutes} min {seconds} sec")
                st.metric("FPS", f"{fps:.2f}")
            st.session_state["video_duration"] = duration_seconds
            cap.release()

    st.divider()

    # SETTINGS SECTION
    st.subheader("⚙️ Recap Settings")
    settings_col1, settings_col2 = st.columns(2)

    with settings_col1:
        whisper_model = st.selectbox("🧠 Whisper Model", ["tiny", "base"], index=0, key="main_whisper_model")
        voice_gender = st.selectbox("🎙️ Voice", ["👩 Female — Nilar", "👨 Male — Thiha"], key="main_voice_gender")
        voice_speed = st.selectbox("🎚️ Voice Speed", [1.0, 1.1, 1.2], index=0, key="main_voice_speed")

    with settings_col2:
        freeze_enabled = st.checkbox("🧊 Enable Freeze Frame + Zoom", value=True, key="main_freeze_enabled")
        freeze_interval = st.number_input("⏱️ Freeze Every", min_value=5.0, max_value=60.0, value=10.0, step=1.0, key="main_freeze_interval")
        freeze_duration = st.number_input("🧊 Freeze Duration", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="main_freeze_duration")

    st.divider()

    # ONE CLICK TRIGGER BUTTON
    if uploaded_file is not None:
        st.markdown("### 🎬 Ready to Generate")
        if st.button("🚀 ONE CLICK — GENERATE MOVIE RECAP", type="primary", use_container_width=True):
            st.session_state["selected_whisper_model"] = whisper_model
            st.session_state["selected_voice_gender"] = voice_gender
            st.session_state["selected_voice_speed"] = voice_speed
            st.session_state["selected_freeze_enabled"] = freeze_enabled
            st.session_state["selected_freeze_interval"] = freeze_interval
            st.session_state["selected_freeze_duration"] = freeze_duration
            
            st.session_state["processing"] = True
            st.session_state["process_percent"] = 0
            st.session_state["process_step"] = "Starting One-Click Flow..."
            st.session_state["process_complete"] = False
            st.rerun()

# =========================================================
# FINAL RESULT DISPLAY (100% COMPLETE SCREEN)
# =========================================================
if st.session_state["process_complete"]:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    st.balloons()
    st.markdown("""
    <div style="text-align:center; padding:20px;">
        <h1 style="color:#10b981 !important;">🎉 100% COMPLETE</h1>
        <h3>Your Movie Recap is Ready!</h3>
    </div>
    """, unsafe_allow_html=True)

    output_video_path = st.session_state["final_output_video"]
    
    if os.path.exists(output_video_path):
        with open(output_video_path, "rb") as v_file:
            video_bytes = v_file.read()
            
        st.subheader("▶️ Preview Final Video")
        st.video(video_bytes)
        
        st.divider()
        st.download_button(
            label="⬇️ Download Final Video",
            data=video_bytes,
            file_name="final_movie_recap.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True
        )

    if st.button("🔄 Create Another Recap"):
        st.session_state["processing"] = False
        st.session_state["process_complete"] = False
        st.session_state["process_percent"] = 0
        st.rerun()
