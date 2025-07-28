# app.py - MoviePy ClipStitcher (SRT-based, Stabilized)
import os
import json
import tempfile
import traceback
import re
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx.all import crop, resize
import gdown
from openai import OpenAI

# ----------
# Helper Functions
# ----------

def get_api_key() -> str:
    """Retrieve the OpenAI API key from Streamlit secrets or environment."""
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    for key in ("OPENAI_API_KEY", "api_key"):
        if key in st.secrets:
            return st.secrets[key]
    return os.getenv("OPENAI_API_KEY", "")

def get_system_prompt() -> str:
    """Returns the system prompt for the AI."""
    return """
You are an expert viral video editor. Your task is to analyze a provided SRT transcript and select the most compelling, high-impact segments that can be stitched together to create a single, cohesive, and viral short video (under 60 seconds).

CRITICAL REQUIREMENTS:
1.  **Stitch Segments:** You are not creating one continuous clip. You must select multiple, separate segments from the transcript that, when combined, tell a story or make a powerful point.
2.  **Total Duration:** The COMBINED duration of all selected segments MUST be between 20 and 59 seconds.
3.  **Narrative Flow:** The final stitched video must have a clear narrative arc:
    - **Hook:** The very first segment you choose must be an incredibly strong hook (a shocking statement, a question, a bold claim).
    - **Middle:** Subsequent segments should build on the hook, providing context, examples, or escalating the idea.
    - **Payoff:** The final segment should provide a conclusion, a punchline, or a call to action.
4.  **Context:** The final video must make sense on its own, without any external context.

The SRT is provided in a simplified format: `[index] | [start_time] --> [end_time] | [text]`

OUTPUT FORMAT:
You must output ONLY a valid JSON object with these three top-level keys: "segments", "reason", and "title".

- "segments": An array of objects, where each object represents a clip to be stitched in order. Each object must have "start", "end", and "text" keys.
- "reason": A brief explanation of why this combination of clips will be effective.
- "title": A catchy, viral title for the final video.

EXAMPLE JSON OUTPUT:
{
  "segments": [
    {
      "start": "00:00:04,439",
      "end": "00:00:06,169",
      "text": "The single biggest mistake that I see people make..."
    },
    {
      "start": "00:01:20,789",
      "end": "00:01:23,949",
      "text": "...is that they focus on the wrong metrics entirely."
    },
    {
      "start": "00:02:55,439",
      "end": "00:02:58,339",
      "text": "And here's how you can fix it in less than 60 seconds."
    }
  ],
  "reason": "This sequence creates a powerful narrative. It starts with a strong hook, presents a problem, and finishes with a clear solution, making it highly engaging and shareable.",
  "title": "The Biggest Mistake You're Making"
}

CRITICAL: Do not include any text, notes, or explanations outside of the JSON object. The entire output must be the JSON itself.
"""

def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,ms) to seconds."""
    try:
        time_parts = time_str.split(',')
        h, m, s = time_parts[0].split(':')
        ms = time_parts[1]
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    except Exception:
        st.error(f"Could not parse SRT time: {time_str}")
        return 0

def parse_srt(srt_content: str) -> (list, str):
    """Parses SRT file content into a list of dictionaries and a simplified text format for the AI."""
    srt_pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', re.S)
    matches = srt_pattern.findall(srt_content)
    segments = []
    simplified_transcript = []
    for match in matches:
        index, start, end, text = match
        text = text.strip().replace('\n', ' ')
        segments.append({"index": int(index), "start": start, "end": end, "text": text})
        simplified_transcript.append(f"[{index}] | {start} --> {end} | {text}")
    return segments, "\n".join(simplified_transcript)

def analyze_srt_transcript(transcript_text: str, client: OpenAI) -> str:
    """Get segment suggestions from AI based on the SRT transcript."""
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": f"Analyze this SRT transcript and identify the best segments to stitch together into a viral clip. Remember, the combined duration must be 20-59 seconds.\n\nTranscript:\n{transcript_text}"}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"AI analysis error: {str(e)}")
        raise

def parse_ai_response(text: str) -> (list, str, str):
    """Parse JSON text from AI into a list of segments."""
    try:
        data = json.loads(text)
        segments = data.get("segments", [])
        reason = data.get("reason", "No reason provided.")
        title = data.get("title", "Untitled Clip")
        if not segments:
            st.warning("AI response did not contain any valid segments to process.")
            return [], reason, title
        valid_segments = [seg for seg in segments if all(key in seg for key in ["start", "end", "text"])]
        return valid_segments, reason, title
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.error(f"Raw text received from AI: {text[:500]}...")
        return [], "JSON parsing failed.", "Error"
    except Exception as e:
        st.error(f"An unexpected error occurred while parsing the AI response: {e}")
        return [], "Parsing failed.", "Error"

def generate_stitched_clip(video_path: str, segments: list, make_vertical: bool = False) -> str:
    """Uses moviepy to cut and stitch video segments together into a single clip."""
    if not segments:
        st.error("Cannot generate video: No segments were provided.")
        return None

    subclips = []
    video = None
    final_clip = None
    try:
        video = VideoFileClip(video_path)
        total_duration = video.duration
        st.info(f"Source video loaded: {video.w}x{video.h}, {total_duration:.1f}s")
        st.info(f"Attempting to stitch {len(segments)} segments...")

        for i, seg in enumerate(segments, start=1):
            start_time = srt_time_to_seconds(seg.get("start"))
            end_time = srt_time_to_seconds(seg.get("end"))
            if start_time >= end_time or start_time > total_duration:
                st.warning(f"Skipping segment {i}: Invalid time range ({start_time:.2f}s -> {end_time:.2f}s).")
                continue
            end_time = min(end_time, total_duration)
            st.write(f"Creating subclip {i}: from {start_time:.2f}s to {end_time:.2f}s")
            subclips.append(video.subclip(start_time, end_time))

        if not subclips:
            st.error("No valid subclips could be created. Aborting.")
            return None

        final_clip = concatenate_videoclips(subclips)

        if make_vertical:
            st.info("Converting final clip to vertical 9:16 format...")
            (w, h) = final_clip.size
            target_aspect_ratio = 9 / 16.0
            crop_width = h * target_aspect_ratio
            if crop_width > w:
                final_clip = crop(final_clip, y_center=h/2, height=w/target_aspect_ratio)
            else:
                final_clip = crop(final_clip, x_center=w/2, width=h*target_aspect_ratio)
            final_clip = resize(final_clip, height=1920)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="stitched_clip_")
        st.info(f"Writing final stitched clip to {temp_file.name}...")
        final_clip.write_videofile(temp_file.name, codec="libx264", audio_codec="aac", threads=4)
        st.success("✅ Final clip generated successfully!")
        return temp_file.name
    except Exception as e:
        st.error(f"Error during clip generation: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # CRITICAL: Close all video objects to free up memory
        if final_clip:
            final_clip.close()
        for sc in subclips:
            sc.close()
        if video:
            video.close()

def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download a Google Drive file given its share URL to out_path."""
    try:
        file_id_match = re.search(r'/file/d/([^/]+)', drive_url) or re.search(r'id=([^&]+)', drive_url)
        if not file_id_match:
            raise ValueError("Could not extract file ID from Google Drive URL. Please ensure it's a valid sharing link.")
        file_id = file_id_match.group(1)
        gdown.download(id=file_id, output=out_path, quiet=False)
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        else:
            raise Exception("Download failed. The file may be private, or the link is incorrect. Please ensure 'Anyone with the link' can view.")
    except Exception as e:
        raise Exception(f"Google Drive Download Error: {str(e)}")

# ----------
# Streamlit App
# ----------

def main():
    st.set_page_config(page_title="SRT ClipStitcher", layout="wide")
    st.title("🎬 SRT ClipStitcher")
    st.markdown("Upload a Google Drive link and its SRT transcript to automatically create a viral short video by stitching the best moments together.")

    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.processing_complete = False
        st.session_state.final_clip_path = None
        st.session_state.clip_recipe = None
        st.session_state.clip_reason = None
        st.session_state.clip_title = None
        st.session_state.video_path = None
        st.session_state.srt_content = None

    API_KEY = get_api_key()
    if not API_KEY:
        st.error("❌ OpenAI API key not found. Add it to Streamlit secrets or set the OPENAI_API_KEY environment variable.")
        return
    client = OpenAI(api_key=API_KEY)

    st.sidebar.header("⚙️ Inputs")
    drive_url = st.sidebar.text_input("🔗 Google Drive Video URL", placeholder="https://drive.google.com/file/d/...")
    srt_file = st.sidebar.file_uploader("📄 Upload SRT File", type=["srt"])
    make_vertical = st.sidebar.checkbox("Create Vertical Clip (9:16)", value=True)

    if drive_url and not st.session_state.video_path:
        with st.spinner("Downloading video from Google Drive..."):
            try:
                tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                st.session_state.video_path = download_drive_file(drive_url, tmp_video.name)
                st.success("✅ Video downloaded successfully!")
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()
    
    if st.session_state.video_path:
        st.video(st.session_state.video_path)

    if srt_file and not st.session_state.srt_content:
        st.session_state.srt_content = srt_file.getvalue().decode("utf-8")
        st.success("✅ SRT file loaded.")
    
    if st.session_state.srt_content:
        with st.expander("View Full Transcript"):
            st.text_area("SRT Content", st.session_state.srt_content, height=200, key="srt_display")

    if not (st.session_state.video_path and st.session_state.srt_content):
        st.info("👋 Welcome! Please provide a Google Drive link and an SRT file in the sidebar to begin.")
        st.stop()

    if st.session_state.processing_complete:
        st.header("🎉 Your Stitched Clip is Ready!")
        if st.session_state.final_clip_path and os.path.isfile(st.session_state.final_clip_path):
            st.subheader(f"🎬 {st.session_state.clip_title}")
            st.video(st.session_state.final_clip_path)
            with open(st.session_state.final_clip_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Your Clip", data=file,
                    file_name=f"{st.session_state.clip_title.replace(' ', '_').lower()}.mp4",
                    mime="video/mp4", use_container_width=True, type="primary"
                )
            st.subheader("💡 AI Analysis & Composition")
            st.info(f"**Reasoning:** {st.session_state.clip_reason}")
            st.warning("This clip was created by stitching the following segments together:")
            for i, segment in enumerate(st.session_state.clip_recipe, 1):
                st.markdown(f"**Part {i}:** `{segment['start']} --> {segment['end']}`")
                st.text(f"\"{segment['text']}\"")
        else:
            st.error("The generated clip file could not be found. Please try generating it again.")
    else:
        if st.button("🚀 Analyze & Create Clip", type="primary", use_container_width=True):
            with st.spinner("Processing... This may take several minutes."):
                try:
                    status_text = st.empty()
                    status_text.text("Parsing SRT file...")
                    _, srt_for_ai = parse_srt(st.session_state.srt_content)
                    status_text.text("🤖 Asking AI to find the best moments...")
                    ai_response_json = analyze_srt_transcript(srt_for_ai, client)
                    status_text.text("✅ AI analysis complete. Parsing response...")
                    segments_to_stitch, reason, title = parse_ai_response(ai_response_json)
                    if not segments_to_stitch:
                        st.error("AI did not return valid segments to stitch. Please try again.")
                        st.stop()
                    status_text.text(f"✂️ Stitching {len(segments_to_stitch)} segments into one video...")
                    final_clip_path = generate_stitched_clip(st.session_state.video_path, segments_to_stitch, make_vertical)
                    if final_clip_path:
                        st.session_state.final_clip_path = final_clip_path
                        st.session_state.clip_recipe = segments_to_stitch
                        st.session_state.clip_reason = reason
                        st.session_state.clip_title = title
                        st.session_state.processing_complete = True
                        st.rerun()
                    else:
                        st.error("❌ Clip generation failed. Check the logs above for details.")
                except Exception as e:
                    st.error(f"An error occurred during the process: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", help="Clear all data and start fresh"):
        for key in list(st.session_state.keys()):
            if key != 'app_initialized':
                # Clean up temp files if they exist in session state
                if '_path' in key and st.session_state[key] and isinstance(st.session_state[key], str) and os.path.isfile(st.session_state[key]):
                    try:
                        os.unlink(st.session_state[key])
                    except: pass
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
