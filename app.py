# app.py - MoviePy ClipStitcher (SRT-based, Fixed)
import os
import json
import tempfile
import traceback
import re
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
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
    return os.getenv("OPENAI_API_KEY", "")

def get_system_prompt() -> str:
    """Returns the system prompt for the AI."""
    return """
You are an expert viral video editor. Your task is to analyze a provided SRT transcript and select the most compelling, high-impact segments that can be stitched together to create a single, cohesive, and viral short video (under 60 seconds).

CRITICAL REQUIREMENTS:
1.  **Stitch Segments:** You are not creating one continuous clip. You must select multiple, separate segments from the transcript that, when combined, tell a story or make a powerful point.
2.  **Total Duration:** The COMBINED duration of all selected segments MUST be between 20 and 59 seconds.
3.  **Narrative Flow:** The final stitched video must have a clear narrative arc: Hook -> Middle -> Payoff.
4.  **Context:** The final video must make sense on its own.

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
    }
  ],
  "reason": "This sequence creates a powerful narrative. It starts with a strong hook and presents a problem.",
  "title": "The Biggest Mistake You're Making"
}

CRITICAL: Do not include any text, notes, or explanations outside of the JSON object.
"""

def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,ms) to seconds."""
    try:
        time_parts = time_str.split(',')
        h, m, s = time_parts[0].split(':')
        ms = time_parts[1] if len(time_parts) > 1 else '0'
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    except Exception as e:
        st.warning(f"Could not parse SRT time: {time_str} - {e}")
        return 0

def parse_srt(srt_content: str) -> tuple:
    """Parses SRT file content."""
    # Fixed regex pattern with proper escaping
    srt_pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n*$)', re.DOTALL)
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
    """Get segment suggestions from AI."""
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": f"Analyze this SRT transcript and identify the best segments to stitch together into a viral clip.\n\nTranscript:\n{transcript_text}"}
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

def parse_ai_response(text: str) -> tuple:
    """Parse JSON text from AI."""
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

def generate_stitched_clip(video_path: str, segments: list, make_vertical: bool, status_text) -> str:
    """Uses moviepy to cut and stitch video segments together."""
    if not segments:
        st.error("Cannot generate video: No segments were provided.")
        return None

    subclips = []
    video = None
    final_clip = None
    
    try:
        status_text.text("Loading source video into memory...")
        video = VideoFileClip(video_path)
        total_duration = video.duration
        st.info(f"Source video loaded: {video.w}x{video.h}, {total_duration:.1f}s")
        
        for i, seg in enumerate(segments, start=1):
            status_text.text(f"Processing segment {i}/{len(segments)}...")
            start_time = srt_time_to_seconds(seg.get("start"))
            end_time = srt_time_to_seconds(seg.get("end"))
            
            if start_time >= end_time or start_time > total_duration:
                st.warning(f"Skipping segment {i}: Invalid time range.")
                continue
                
            end_time = min(end_time, total_duration)
            subclip = video.subclip(start_time, end_time)
            subclips.append(subclip)

        if not subclips:
            st.error("No valid subclips could be created. Aborting.")
            return None

        status_text.text("Stitching segments together...")
        final_clip = concatenate_videoclips(subclips)

        if make_vertical:
            status_text.text("Converting to vertical 9:16 format...")
            w, h = final_clip.size
            target_aspect = 9.0 / 16.0
            clip_aspect = float(w) / h

            if clip_aspect > target_aspect:  # Wider than target
                new_width = int(h * target_aspect)
                final_clip = crop(final_clip, width=new_width, x_center=w/2)
            else:  # Taller than target
                new_height = int(w / target_aspect)
                final_clip = crop(final_clip, height=new_height, y_center=h/2)
            
            final_clip = resize(final_clip, height=1920)

        # Create temporary file with proper cleanup
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="stitched_clip_")
        temp_file.close()  # Close the file handle so moviepy can write to it
        
        status_text.text(f"Writing final video file... (this can take a moment)")
        final_clip.write_videofile(
            temp_file.name, 
            codec="libx264", 
            audio_codec="aac", 
            threads=2, 
            preset='medium',
            verbose=False,
            logger=None
        )
        
        st.success("✅ Final clip generated successfully!")
        return temp_file.name
        
    except Exception as e:
        st.error(f"Error during clip generation: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # CRITICAL: Close all video objects to free up memory
        try:
            if video:
                video.close()
            if final_clip:
                final_clip.close()
            for sc in subclips:
                sc.close()
        except Exception as e:
            st.warning(f"Error cleaning up video objects: {e}")

def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download a Google Drive file."""
    try:
        # Extract file ID from various Google Drive URL formats
        file_id_match = (
            re.search(r'/file/d/([a-zA-Z0-9_-]+)', drive_url) or 
            re.search(r'id=([a-zA-Z0-9_-]+)', drive_url) or
            re.search(r'open\?id=([a-zA-Z0-9_-]+)', drive_url)
        )
        
        if not file_id_match:
            raise ValueError("Could not extract file ID from Google Drive URL.")
            
        file_id = file_id_match.group(1)
        gdown.download(id=file_id, output=out_path, quiet=False)
        
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        else:
            raise Exception("Download failed. The file may be private or link is incorrect.")
            
    except Exception as e:
        raise Exception(f"Google Drive Download Error: {str(e)}")

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="SRT ClipStitcher", layout="wide")
    st.title("🎬 SRT ClipStitcher")
    st.markdown("Upload a Google Drive link and its SRT transcript to create a viral short video.")

    # Initialize state
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            "video_path": None,
            "srt_content": None,
            "final_clip_path": None,
            "clip_recipe": None,
            "clip_reason": None,
            "clip_title": None,
            "processing_complete": False
        }
    
    state = st.session_state.app_state

    # Get API key
    api_key = get_api_key()
    if not api_key:
        st.error("❌ OpenAI API key is missing. Please check your Streamlit secrets or environment variables.")
        st.stop()
    
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI client: {e}")
        st.stop()

    # --- Sidebar for Inputs ---
    st.sidebar.header("⚙️ Inputs")
    drive_url = st.sidebar.text_input("🔗 Google Drive Video URL", placeholder="https://drive.google.com/file/d/...")
    srt_file = st.sidebar.file_uploader("📄 Upload SRT File", type=["srt"])
    make_vertical = st.sidebar.checkbox("Create Vertical Clip (9:16)", value=True)

    # --- Input Handling ---
    if drive_url and not state["video_path"]:
        with st.spinner("Downloading video from Google Drive..."):
            try:
                tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_video.close()  # Close file handle
                state["video_path"] = download_drive_file(drive_url, tmp_video.name)
                st.success("✅ Video downloaded!")
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()
    
    if srt_file and not state["srt_content"]:
        try:
            state["srt_content"] = srt_file.getvalue().decode("utf-8")
            st.success("✅ SRT file loaded.")
        except Exception as e:
            st.error(f"❌ Error reading SRT file: {e}")

    # --- Main Screen Logic ---
    if not (state["video_path"] and state["srt_content"]):
        st.info("👋 Welcome! Please provide a Google Drive link and an SRT file in the sidebar to begin.")
        st.stop()

    if state["video_path"] and os.path.isfile(state["video_path"]):
        st.video(state["video_path"])
    
    if state["srt_content"]:
        with st.expander("View Full Transcript"):
            st.text_area("SRT Content", state["srt_content"], height=200, key="srt_display")

    # --- Processing and Display ---
    if state["processing_complete"]:
        st.header("🎉 Your Stitched Clip is Ready!")
        if state["final_clip_path"] and os.path.isfile(state["final_clip_path"]):
            st.subheader(f"🎬 {state['clip_title']}")
            st.video(state["final_clip_path"])
            
            with open(state["final_clip_path"], "rb") as file:
                st.download_button(
                    label="⬇️ Download Your Clip", 
                    data=file,
                    file_name=f"{state['clip_title'].replace(' ', '_').lower()}.mp4",
                    mime="video/mp4", 
                    use_container_width=True, 
                    type="primary"
                )
            
            st.subheader("💡 AI Analysis & Composition")
            st.info(f"**Reasoning:** {state['clip_reason']}")
            st.warning("This clip was created by stitching the following segments:")
            
            for i, segment in enumerate(state['clip_recipe'], 1):
                st.markdown(f"**Part {i}:** `{segment['start']} --> {segment['end']}`")
                st.text(f"\"{segment['text']}\"")
        else:
            st.error("The generated clip file could not be found. Please try generating it again.")
    else:
        if st.button("🚀 Analyze & Create Clip", type="primary", use_container_width=True):
            status_text = st.empty()
            
            with st.spinner("Processing... This may take several minutes."):
                try:
                    status_text.text("Parsing SRT file...")
                    _, srt_for_ai = parse_srt(state["srt_content"])
                    
                    if not srt_for_ai.strip():
                        st.error("❌ Could not parse SRT file. Please check the format.")
                        st.stop()
                    
                    status_text.text("🤖 Asking AI to find the best moments...")
                    ai_response_json = analyze_srt_transcript(srt_for_ai, client)
                    
                    status_text.text("✅ AI analysis complete. Parsing response...")
                    segments_to_stitch, reason, title = parse_ai_response(ai_response_json)
                    
                    if not segments_to_stitch:
                        st.error("AI did not return valid segments to stitch. Please try again.")
                        st.stop()

                    final_clip_path = generate_stitched_clip(
                        state["video_path"], 
                        segments_to_stitch, 
                        make_vertical, 
                        status_text
                    )
                    
                    if final_clip_path:
                        state["final_clip_path"] = final_clip_path
                        state["clip_recipe"] = segments_to_stitch
                        state["clip_reason"] = reason
                        state["clip_title"] = title
                        state["processing_complete"] = True
                        st.rerun()
                    else:
                        st.error("❌ Clip generation failed. Check the logs above for details.")
                        
                except Exception as e:
                    st.error(f"An error occurred during the process: {e}")
                    traceback.print_exc()

    # --- Reset Button ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", help="Clear all data and start fresh"):
        # Clean up temp files before clearing state
        for key, value in state.items():
            if '_path' in key and value and isinstance(value, str) and os.path.isfile(value):
                try: 
                    os.unlink(value)
                except: 
                    pass
        
        # Reset state
        st.session_state.app_state = {
            "video_path": None, 
            "srt_content": None, 
            "final_clip_path": None,
            "clip_recipe": None, 
            "clip_reason": None, 
            "clip_title": None,
            "processing_complete": False
        }
        st.rerun()

if __name__ == "__main__":
    main()
