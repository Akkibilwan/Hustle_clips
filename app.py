# Exact Copy of Working Code Structure - SRT ClipStitcher
import os
import re
import tempfile
import streamlit as st
import traceback
import shutil
import gc

# All necessary libraries exactly like working code
from moviepy.editor import VideoFileClip, concatenate_videoclips
import gdown
from openai import OpenAI

# ---
# 1. SYSTEM PROMPT (Using working code approach)
# ---
SYSTEM_PROMPT = """
You are an expert YouTube Shorts strategist and video editor.

Your job is to analyze the full transcript of a long-form interview or podcast and extract powerful 30–60 second Shorts using two formats:
1. Direct Clips — continuous timestamp segments that tell a complete story.
2. Franken-Clips — stitched from non-contiguous timestamps, using a hook from one part and payoff from another.

---

🛑 STRICT RULE: DO NOT REWRITE OR SUMMARIZE ANY DIALOGUE.

You must:
- Use the transcript lines exactly as they appear in the provided SRT/transcript.
- Do not shorten, reword, paraphrase, or compress the speaker's sentences.
- Keep all original punctuation, phrasing, and spelling.
- Only include full dialogue blocks — no cherry-picking fragments from within a block.
- ALWAYS provide EXACT timestamps in HH:MM:SS,mmm format (e.g., 00:01:23,450)

The output should allow a video editor to directly cut the clip using the given timestamps and script.

---

📦 OUTPUT FORMAT (repeat for each Short):

**Short Title:** [Catchy title with emoji]
**Estimated Duration:** [e.g., 42 seconds]
**Type:** [Direct Clip / Franken-Clip]

**Timestamps:**
START: 00:01:23,450 --> END: 00:01:35,200
[For Franken-clips, list multiple timestamp ranges]

**Script:**
[Exact dialogue from transcript - no modifications]

**Rationale:**
[Brief explanation why this will go viral]

---

🛑 CRITICAL REMINDERS:
- Provide EXACT timestamps that match the SRT format
- Do not modify any dialogue
- Ensure timestamps are accurate and complete
- Each clip should be 30-60 seconds total

Generate the requested number of shorts following this exact format.
"""

# ---
# 2. HELPER FUNCTIONS (Exact copy from working code)
# ---

def get_openai_api_key() -> str:
    return st.secrets.get("openai", {}).get("api_key", "")

def read_transcript_file(uploaded_file) -> str:
    try:
        return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def parse_srt_timestamp(timestamp_str: str) -> float:
    """Convert SRT timestamp format to total seconds."""
    timestamp_str = timestamp_str.strip().replace(',', '.')
    try:
        time_parts = timestamp_str.split(':')
        if len(time_parts) == 3:
            h, m, s_ms = time_parts
            return int(h) * 3600 + int(m) * 60 + float(s_ms)
        elif len(time_parts) == 2:
            m, s_ms = time_parts
            return int(m) * 60 + float(s_ms)
        return float(time_parts[0])
    except Exception:
        return 0.0

def analyze_transcript_with_llm(transcript: str, count: int):
    user_content = f"{transcript}\n\nPlease generate {count} unique potential shorts following the exact format specified."
    
    api_key = get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key not set.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            temperature=0.7, max_tokens=4000
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

def parse_ai_output(text: str) -> list:
    clips = []
    sections = re.split(r'\*\*Short Title:\*\*', text)
    
    for i, section in enumerate(sections[1:], 1):
        try:
            title_match = re.search(r'^(.*?)(?:\n|\*\*)', section, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else f"Untitled Clip {i}"
            
            type_match = re.search(r'\*\*Type:\*\*\s*(.*?)(?:\n|\*\*)', section)
            clip_type = type_match.group(1).strip() if type_match else "Unknown"

            rationale_match = re.search(r'\*\*Rationale:\*\*(.*?)(?:\n\*\*|$)', section, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

            script_match = re.search(r'\*\*Script:\*\*(.*?)(?=\*\*Rationale:\*\*)', section, re.DOTALL)
            script = script_match.group(1).strip() if script_match else "Script not found."

            timestamp_text_match = re.search(r'\*\*Timestamps:\*\*(.*?)(?=\*\*Script:\*\*)', section, re.DOTALL)
            timestamps = []
            if timestamp_text_match:
                timestamp_text = timestamp_text_match.group(1)
                timestamp_matches = re.findall(r'START:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*END:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', timestamp_text)
                for start_str, end_str in timestamp_matches:
                    start_sec = parse_srt_timestamp(start_str)
                    end_sec = parse_srt_timestamp(end_str)
                    timestamps.append({"start_str": start_str, "end_str": end_str, "start_sec": start_sec, "end_sec": end_sec})

            if timestamps:
                clips.append({
                    "title": title, "type": clip_type, "rationale": rationale,
                    "script": script, "timestamps": timestamps
                })
        except Exception as e:
            st.warning(f"Could not parse clip section {i}: {e}")
    return clips

def download_drive_file(drive_url: str, download_path: str) -> str:
    """Downloads a Google Drive file and verifies its integrity - EXACT COPY."""
    try:
        output_path = os.path.join(download_path, 'downloaded_video.mp4')
        gdown.download(drive_url, output_path, quiet=False, fuzzy=True)

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            raise Exception("Downloaded file is missing or empty.")

        try:
            with VideoFileClip(output_path) as clip:
                duration = clip.duration
            if duration is None or duration <= 0:
                raise Exception("Video file is corrupted (duration is zero or None).")
            st.info(f"Verified downloaded file. Duration: {duration:.2f} seconds.")
            return output_path
        except Exception as e:
            raise Exception(f"Downloaded file appears to be corrupted and cannot be read by MoviePy. Error: {e}. This often happens with incomplete downloads. Please check the Google Drive sharing settings and try again.")

    except Exception as e:
        raise Exception(f"Google Drive download failed: {e}. Ensure the link is public and correct.")

def generate_clips_progressively(video_path: str, clips_data: list, output_dir: str):
    """
    Generator function that cuts, stitches, and YIELDS video clips one by one - EXACT COPY.
    """
    source_video = VideoFileClip(video_path)
    video_duration = source_video.duration
    
    for i, clip_data in enumerate(clips_data):
        st.info(f"Processing Clip {i+1}/{len(clips_data)}: '{clip_data['title']}'")
        try:
            subclips = []
            for ts in clip_data["timestamps"]:
                start_time, end_time = ts['start_sec'], ts['end_sec']
                if start_time < video_duration and end_time <= video_duration:
                    subclips.append(source_video.subclip(start_time, end_time))
                else:
                    st.warning(f"Segment {ts['start_str']} -> {ts['end_str']} is out of video bounds. Skipping.")
            
            if not subclips:
                st.error(f"No valid segments for clip '{clip_data['title']}'. Skipping.")
                continue

            final_clip = concatenate_videoclips(subclips) if len(subclips) > 1 else subclips[0]
            
            safe_title = re.sub(r'[^\w\s-]', '', clip_data['title']).strip().replace(' ', '_')
            output_filepath = os.path.join(output_dir, f"clip_{i+1}_{safe_title[:20]}.mp4")
            
            final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac", temp_audiofile=f'temp-audio_{i}.m4a', remove_temp=True, logger=None)
            
            # YIELD the completed clip's data - EXACT COPY
            yield {
                "path": output_filepath,
                "title": clip_data['title'],
                "type": clip_data['type'],
                "rationale": clip_data['rationale'],
                "script": clip_data['script'],
                "timestamps": clip_data['timestamps']
            }
            st.success(f"✅ Generated clip: {clip_data['title']}")

        except Exception as e:
            st.error(f"Failed to generate clip '{clip_data['title']}': {e}")
        finally:
            if 'final_clip' in locals(): final_clip.close()
            if 'subclips' in locals():
                for sc in subclips: sc.close()

    source_video.close()

# ---
# 3. STREAMLIT APP - EXACT STRUCTURE FROM WORKING CODE
# ---

def main():
    st.set_page_config(page_title="SRT ClipStitcher", layout="wide", page_icon="🎬")
    
    st.title("🎬 SRT ClipStitcher")
    st.markdown("**Generate video clips from Google Drive videos using SRT analysis.**")

    # Initialize session state - EXACT COPY
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar Configuration - SIMPLIFIED LIKE WORKING CODE
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        video_url = st.text_input("Google Drive URL", placeholder="Paste your Google Drive URL here...")
        uploaded_transcript = st.file_uploader("Upload SRT/Transcript File", type=["srt", "txt"])
        clips_count = st.slider("Number of Clips to Generate:", 1, 10, 3)
    
    # Main action button - EXACT COPY LOGIC
    if st.button("🚀 Generate Video Clips", type="primary", use_container_width=True):
        st.session_state.results = None # Clear previous results
        if not video_url or not uploaded_transcript:
            st.error("❌ Please provide both a video URL and a transcript file.")
            return

        with st.spinner("📖 Reading transcript..."):
            transcript_content = read_transcript_file(uploaded_transcript)
            if not transcript_content: return
        st.success("✅ Transcript loaded.")

        with st.spinner("🧠 Analyzing transcript with AI..."):
            ai_response = analyze_transcript_with_llm(transcript_content, clips_count)
            if not ai_response: return
        st.success("✅ AI analysis complete.")

        with st.spinner("📝 Parsing AI recommendations..."):
            clips_data = parse_ai_output(ai_response)
            if not clips_data:
                st.error("❌ Could not parse any valid clips from the AI response.")
                return
        st.success(f"✅ Parsed {len(clips_data)} recommendations.")
        
        # --- EXACT COPY OF WORKING CODE LOGIC ---
        st.session_state.results = {"type": "generator", "data": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with st.spinner("⬇️ Downloading video from Google Drive..."):
                    video_path = download_drive_file(video_url, temp_dir)
                st.success("✅ Video downloaded and verified.")

                st.markdown("---")
                st.header("🎬 Your Generated Clips (Loading...)")
                
                persistent_dir = "generated_clips"
                if not os.path.exists(persistent_dir):
                    os.makedirs(persistent_dir)
                for f in os.listdir(persistent_dir):
                    os.remove(os.path.join(persistent_dir, f))
                
                # --- EXACT COPY: Progressive Loading Loop ---
                final_clips = []
                clip_generator = generate_clips_progressively(video_path, clips_data, temp_dir)
                for clip in clip_generator:
                    # Move file and display immediately - EXACT COPY
                    new_path = os.path.join(persistent_dir, os.path.basename(clip['path']))
                    shutil.move(clip['path'], new_path)
                    clip['path'] = new_path
                    final_clips.append(clip)
                    
                    # Display the clip that was just generated - EXACT COPY
                    st.subheader(f"🎬 {clip['title']}")
                    col_video, col_info = st.columns(2)
                    with col_video:
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as file:
                            st.download_button("⬇️ Download Clip", file, file_name=os.path.basename(clip['path']), key=f"dl_{new_path}")
                    with col_info:
                        st.markdown(f"**Type:** `{clip['type']}`")
                        with st.expander("💡 Rationale"):
                            st.info(clip['rationale'])
                        with st.expander("📜 Script"):
                            st.text_area("", clip['script'], height=100, key=f"script_{new_path}")
                        with st.expander("⏰ Timestamps Used"):
                            for ts in clip['timestamps']:
                                st.code(f"{ts['start_str']} --> {ts['end_str']}")
                    st.markdown("---")

                st.session_state.results["data"] = final_clips # Save final list
                st.success("🎉 All clips generated!")

            except Exception as e:
                st.error(f"An error occurred during the clip generation process: {e}")
                st.code(traceback.format_exc())

    # Display results from session state if they exist (for reruns) - EXACT COPY
    elif st.session_state.results:
        results = st.session_state.results
        st.markdown("---")
        
        if results["type"] == "generator":
            st.header("✅ Your Generated Clips")
            if not results["data"]:
                st.warning("No clips were successfully generated.")
            for clip in results["data"]:
                st.subheader(f"🎬 {clip['title']}")
                col_video, col_info = st.columns(2)
                with col_video:
                    if os.path.exists(clip['path']):
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as file:
                            st.download_button("⬇️ Download Clip", file, file_name=os.path.basename(clip['path']), key=f"dl_{clip['path']}")
                    else:
                        st.error("Clip file not found.")
                with col_info:
                     st.markdown(f"**Type:** `{clip['type']}`")
                     with st.expander("💡 Rationale"):
                         st.info(clip['rationale'])
                     with st.expander("📜 Script"):
                         st.text_area("", clip['script'], height=100, key=f"script_{clip['path']}")
                     with st.expander("⏰ Timestamps Used"):
                         for ts in clip['timestamps']:
                             st.code(f"{ts['start_str']} --> {ts['end_str']}")
                st.markdown("---")

if __name__ == "__main__":
    main()
