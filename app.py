# app.py - MoviePy ClipMaker (SRT-based stitching)
import os
import json
import tempfile
import traceback
import re
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import crop
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
    """
    Returns the system prompt for the AI.
    This prompt instructs the AI to act as a viral video editor,
    analyzing an SRT transcript to find and combine the best segments.
    """
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

Your task is to analyze the SRT file content below and return a JSON object that specifies which segments to use.

The SRT is provided in a simplified format: `[index] | [start_time] --> [end_time] | [text]`

OUTPUT FORMAT:
You must output ONLY a valid JSON object with a single key, "segments". This key should contain an array of objects, where each object represents a clip to be stitched in order.

Each object in the "segments" array must have these exact keys:
- "start": The exact start timestamp (e.g., "00:01:15,320") from the SRT segment you chose.
- "end": The exact end timestamp (e.g., "00:01:18,120") from the SRT segment you chose.
- "text": The exact text content from that SRT segment.

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
    """
    Parses SRT file content into a list of dictionaries and a simplified text format for the AI.
    """
    srt_pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n', re.S)
    matches = srt_pattern.findall(srt_content)

    segments = []
    simplified_transcript = []
    for match in matches:
        index, start, end, text = match
        text = text.strip().replace('\n', ' ')
        segments.append({
            "index": int(index),
            "start": start,
            "end": end,
            "text": text
        })
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
            response_format={"type": "json_object"} # Enforce JSON output
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"AI analysis error: {str(e)}")
        raise

def parse_ai_response(text: str) -> (list, str, str):
    """Parse JSON text from AI into a list of segments."""
    try:
        # Clean the text in case there's extra content
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        segments = data.get("segments", [])
        reason = data.get("reason", "No reason provided.")
        title = data.get("title", "Untitled Clip")

        if not segments:
            st.warning("AI response did not contain any valid segments to process.")
            return [], reason, title

        # Validate segments
        valid_segments = []
        for i, seg in enumerate(segments):
            if all(key in seg for key in ["start", "end", "text"]):
                valid_segments.append(seg)
            else:
                st.warning(f"Skipping invalid segment {i+1}: Missing required fields")

        return valid_segments, reason, title
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        st.error(f"Raw text received from AI: {text[:500]}...")
        return [], "JSON parsing failed.", "Error"
    except Exception as e:
        st.error(f"An unexpected error occurred while parsing the AI response: {e}")
        return [], "Parsing failed.", "Error"


def generate_stitched_clip(video_path: str, segments: list, make_vertical: bool = False) -> str:
    """
    Uses moviepy to cut and stitch video segments together into a single clip.
    """
    subclips = []
    if not segments:
        st.error("Cannot generate video: No segments were provided.")
        return None

    try:
        # Load video once
        video = VideoFileClip(video_path)
        total_duration = video.duration

        st.info(f"Source video loaded: {video.w}x{video.h}, {total_duration:.1f}s")
        st.info(f"Attempting to stitch {len(segments)} segments...")

        for i, seg in enumerate(segments, start=1):
            start_time = srt_time_to_seconds(seg.get("start"))
            end_time = srt_time_to_seconds(seg.get("end"))

            # Basic validation
            if start_time >= end_time or start_time > total_duration:
                st.warning(f"Skipping segment {i}: Invalid time range ({start_time:.2f}s -> {end_time:.2f}s).")
                continue

            # Ensure end time does not exceed video duration
            end_time = min(end_time, total_duration)

            st.write(f"Creating subclip {i}: from {start_time:.2f}s to {end_time:.2f}s")
            subclip = video.subclip(start_time, end_time)
            subclips.append(subclip)

        if not subclips:
            st.error("No valid subclips could be created. Aborting.")
            video.close()
            return None

        # Concatenate all the valid subclips
        final_clip = concatenate_videoclips(subclips)

        # Optional: Convert to vertical 9:16 format
        if make_vertical:
            st.info("Converting final clip to vertical 9:16 format...")
            (w, h) = final_clip.size
            target_aspect_ratio = 9 / 16
            
            # Crop to 9:16, centered
            crop_width = h * target_aspect_ratio
            if crop_width > w: # If original is already taller than 9:16
                crop_width = w
                crop_height = w / target_aspect_ratio
                x_center = 0
                y_center = (h - crop_height) / 2
            else:
                crop_height = h
                x_center = (w - crop_width) / 2
                y_center = 0

            final_clip = crop(final_clip, width=crop_width, height=crop_height, x_center=x_center, y_center=y_center)
            # Resize to a standard 1080x1920 for social media
            final_clip = final_clip.resize(height=1920)


        # Create temporary file for the final stitched video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="stitched_clip_")
        st.info(f"Writing final stitched clip to {temp_file.name}...")

        # Write the final video file with audio
        final_clip.write_videofile(temp_file.name, codec="libx264", audio_codec="aac")

        # Clean up memory
        for sc in subclips:
            sc.close()
        final_clip.close()
        video.close()
        
        st.success("✅ Final clip generated successfully!")
        return temp_file.name

    except Exception as e:
        st.error(f"Error during clip generation: {str(e)}")
        # Clean up video object if it exists
        if 'video' in locals() and video:
            video.close()
        return None

def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download a Google Drive file given its share URL to out_path."""
    try:
        file_id_match = re.search(r'/file/d/([^/]+)', drive_url) or re.search(r'id=([^&]+)', drive_url)
        if not file_id_match:
            raise ValueError("Could not extract file ID from Google Drive URL. Please ensure it's a valid sharing link.")
        
        file_id = file_id_match.group(1)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        result = gdown.download(download_url, out_path, quiet=False)
        
        if result and os.path.isfile(result) and os.path.getsize(result) > 0:
            return result
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

    # Initialize session state
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.processing_complete = False
        st.session_state.final_clip_path = None
        st.session_state.clip_recipe = None
        st.session_state.clip_reason = None
        st.session_state.clip_title = None

    # Load & validate API Key
    API_KEY = get_api_key()
    if not API_KEY:
        st.error("❌ OpenAI API key not found. Add it to Streamlit secrets or set the OPENAI_API_KEY environment variable.")
        return
    
    try:
        client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI client: {str(e)}")
        return

    # --- Sidebar for Inputs ---
    st.sidebar.header("⚙️ Inputs")

    # 1. Google Drive Link
    st.sidebar.subheader("1. Video Source")
    drive_url = st.sidebar.text_input(
        "🔗 Google Drive Video URL",
        placeholder="https://drive.google.com/file/d/...",
        help="Provide the public sharing link for your video file."
    )

    # 2. SRT File Upload
    st.sidebar.subheader("2. Transcript")
    srt_file = st.sidebar.file_uploader(
        "📄 Upload SRT File",
        type=["srt"],
        help="Upload the corresponding .srt transcript for your video."
    )
    
    # 3. Output format
    st.sidebar.subheader("3. Output Format")
    make_vertical = st.sidebar.checkbox(
        "Create Vertical Clip (9:16)",
        value=True,
        help="Convert the final clip to a 9:16 aspect ratio for social media."
    )


    # --- Main App Logic ---
    video_path = None
    srt_content = None

    if drive_url and 'video_path' not in st.session_state:
        with st.spinner("Downloading video from Google Drive..."):
            try:
                tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                downloaded_path = download_drive_file(drive_url, tmp_video.name)
                st.session_state['video_path'] = downloaded_path
                st.success(f"✅ Video downloaded successfully!")
                st.video(downloaded_path)
            except Exception as e:
                st.error(f"❌ {e}")
                return
    
    if srt_file and 'srt_content' not in st.session_state:
        srt_content = srt_file.getvalue().decode("utf-8")
        st.session_state['srt_content'] = srt_content
        st.success("✅ SRT file loaded.")
        with st.expander("View Full Transcript"):
            st.text_area("SRT Content", srt_content, height=200)

    # Use data from session state if available
    video_path = st.session_state.get('video_path')
    srt_content = st.session_state.get('srt_content')

    if not (video_path and srt_content):
        st.info("👋 Welcome! Please provide a Google Drive link and an SRT file in the sidebar to begin.")
        return

    # --- Processing and Display ---
    if st.session_state.processing_complete:
        st.header("🎉 Your Stitched Clip is Ready!")
        
        final_clip_path = st.session_state.final_clip_path
        clip_recipe = st.session_state.clip_recipe
        clip_reason = st.session_state.clip_reason
        clip_title = st.session_state.clip_title

        if final_clip_path and os.path.isfile(final_clip_path):
            st.subheader(f"🎬 {clip_title}")
            st.video(final_clip_path)

            with open(final_clip_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Your Clip",
                    data=file,
                    file_name=f"{clip_title.replace(' ', '_').lower()}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary"
                )

            st.markdown("---")
            st.subheader("💡 AI Analysis")
            st.info(f"**Reasoning:** {clip_reason}")

            st.subheader("📜 Clip Composition")
            st.warning("This clip was created by stitching the following segments together:")
            for i, segment in enumerate(clip_recipe, 1):
                st.markdown(f"**Part {i}:** `{segment['start']} --> {segment['end']}`")
                st.text(f"\"{segment['text']}\"")
        else:
            st.error("The generated clip file could not be found. It might have been cleared. Please try generating it again.")

    else: # If not yet processed
        if st.button("🚀 Analyze & Create Clip", type="primary", use_container_width=True):
            with st.spinner("Processing... This may take several minutes."):
                try:
                    # 1. Parse SRT
                    status_text = st.empty()
                    status_text.text("Parsing SRT file...")
                    srt_segments, srt_for_ai = parse_srt(srt_content)
                    if not srt_segments:
                        st.error("Could not parse SRT file. Please check its format.")
                        return

                    # 2. AI Analysis
                    status_text.text("🤖 Asking AI to find the best moments...")
                    ai_response_json = analyze_srt_transcript(srt_for_ai, client)
                    
                    # 3. Parse AI Response
                    status_text.text("✅ AI analysis complete. Parsing response...")
                    segments_to_stitch, reason, title = parse_ai_response(ai_response_json)
                    if not segments_to_stitch:
                        st.error("AI did not return valid segments to stitch. Please try again.")
                        return

                    # 4. Generate Video
                    status_text.text(f"✂️ Stitching {len(segments_to_stitch)} segments into one video...")
                    final_clip_path = generate_stitched_clip(video_path, segments_to_stitch, make_vertical)

                    if final_clip_path:
                        st.session_state.final_clip_path = final_clip_path
                        st.session_state.clip_recipe = segments_to_stitch
                        st.session_state.clip_reason = reason
                        st.session_state.clip_title = title
                        st.session_state.processing_complete = True
                        st.rerun()
                    else:
                        st.error("❌ Clip generation failed. Please check the logs above for details.")

                except Exception as e:
                    st.error(f"An error occurred during the process: {e}")
                    traceback.print_exc()

    # --- Reset Button ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over", help="Clear all data and start fresh"):
        # Clean up any temp files
        if 'video_path' in st.session_state and st.session_state['video_path'] and os.path.isfile(st.session_state['video_path']):
            try:
                os.unlink(st.session_state['video_path'])
            except: pass
        if 'final_clip_path' in st.session_state and st.session_state['final_clip_path'] and os.path.isfile(st.session_state['final_clip_path']):
            try:
                os.unlink(st.session_state['final_clip_path'])
            except: pass

        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
