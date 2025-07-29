
import os
import re
import tempfile
import streamlit as st
import traceback
import shutil

# All necessary libraries exactly like working code
from moviepy.editor import VideoFileClip, concatenate_videoclips
import gdown
from openai import OpenAI

# ---
# 1. SYSTEM PROMPT - "PRECISION EDITOR" LOGIC
# ---
SYSTEM_PROMPT = """
You are an expert viral video editor specializing in "Franken-Clips". Your task is to create a compelling 25-60 second story by stitching together non-contiguous clips.

**CRITICAL INSTRUCTION: THE INPUT FORMAT**
You will receive a highly granular, 'word-level' SRT transcript. Each numbered line may only be a word or a short phrase. Your primary task is to group these small lines together to form longer, meaningful "Logical Segments".

---

**🔥 YOUR 3-STEP EDITING PROCESS:**

**STEP 1: Identify a "Logical Segment"**
- A "Logical Segment" is a complete sentence or a powerful idea.
- To create one, you must **group together several consecutive granular SRT lines.**
- For example, you might group lines #10, #11, and #12 because together they form a great hook.

**STEP 2: Determine the Precise Timestamp for your Logical Segment**
- Once you have grouped your granular lines, you must create a single, combined timestamp.
- Use the **START time of the FIRST granular line** in your group.
- Use the **END time of the LAST granular line** in your group.
- **Example:** If you group lines 10, 11, and 12, your final timestamp is `[START_TIME_of_10] --> [END_TIME_of_12]`.

**STEP 3: Build the Franken-Clip**
- Repeat this process to create 3-4 "Logical Segments" from DIFFERENT parts of the transcript (2+ minutes apart).
- Arrange them into a **HOOK → BUILD → PAYOFF** narrative structure.
- Prioritize viral themes: Vulnerability, Money/Career, Transformations, Industry Secrets, etc.

---

📦 STRICT OUTPUT FORMAT (Use the combined timestamps you created):

**Short Title:** [🚀 Viral Title with an Emoji]
**Theme Category:** [Vulnerability/Money/Transformation/Industry/Advice/Norms]
**Number of Segments:** [3 or 4]

**Selected Segments:**
SEGMENT 1: 00:01:23,450 --> 00:01:28,100 - This is my hook, created by merging lines 25-29. [HOOK]
SEGMENT 2: 00:08:45,100 --> 00:08:52,500 - This builds the story, created from lines 150-155. [BUILD]
SEGMENT 3: 00:25:10,300 --> 00:25:19,900 - The amazing payoff, created from lines 412-418. [PAYOFF]

**Coherence Validation:**
- Strong hook: YES - It creates powerful curiosity.
- Logical flow: YES - The story builds perfectly.
- Satisfying payoff: YES - It delivers a strong emotional or intellectual reward.
- APPROVED: YES

**Viral Strategy:**
[Explain why grouping these specific granular phrases from different parts of the video creates a unique and powerful narrative that wasn't obvious before.]

---

🛑 REQUIREMENTS:
- **You MUST group granular lines** and use the start-time-of-first and end-time-of-last for your timestamps.
- Ensure the final combined duration is 25-60 seconds.
- You MUST select segments far apart in the original video.
- You MUST include "APPROVED: YES" in your validation.

Now, analyze the provided granular transcript and generate 2-3 unique Franken-Clips.
"""

# ---
# 2. HELPER FUNCTIONS
# ---

def get_openai_api_key() -> str:
    """Gets the OpenAI API key from Streamlit secrets."""
    return st.secrets.get("openai", {}).get("api_key", "")

# NEW: Function to fetch available models from OpenAI
def fetch_openai_models(api_key: str) -> list[str]:
    """Fetches available chat models from the OpenAI API."""
    default_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    if not api_key:
        st.warning("OpenAI API key not found. Using default model list.")
        return default_models
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        # Filter for chat models and prioritize the most common/powerful ones
        chat_models = sorted([
            model.id for model in models 
            if "gpt" in model.id and "instruct" not in model.id and "/" not in model.id
        ], reverse=True)
        
        # Ensure priority models are at the top if they exist
        priority_list = [m for m in default_models if m in chat_models]
        other_models = [m for m in chat_models if m not in default_models]
        
        return priority_list + other_models
    except Exception as e:
        st.warning(f"Could not fetch OpenAI models due to an API error: {e}. Using default list.")
        return default_models

def parse_srt_timestamp(timestamp_str: str) -> float:
    """Convert SRT timestamp format to total seconds."""
    timestamp_str = timestamp_str.strip().replace(',', '.')
    try:
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            h, m, s_ms = parts
            return int(h) * 3600 + int(m) * 60 + float(s_ms)
        elif len(parts) == 2:
            m, s_ms = parts
            return int(m) * 60 + float(s_ms)
        return float(parts[0])
    except (ValueError, IndexError):
        st.warning(f"Could not parse timestamp: {timestamp_str}")
        return 0.0

def read_transcript_file(uploaded_file) -> str:
    """Reads the raw transcript file content."""
    try:
        content = uploaded_file.read().decode("utf-8")
        st.info("✅ Granular transcript loaded. The AI will now group the best phrases.")
        return content
    except Exception as e:
        st.error(f"Error reading transcript file: {e}")
        return ""

# MODIFIED: Accepts a 'model' parameter
def analyze_transcript_with_llm(transcript: str, count: int, model: str):
    """Analyzes the transcript with the specified AI model."""
    user_content = f"Here is the granular, word-level transcript:\n\n{transcript}\n\nPlease generate {count} unique Franken-Clips by grouping these lines and following all instructions."
    
    api_key = get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key not set in Streamlit secrets.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model, # Use the selected model
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            temperature=0.7,
            max_tokens=4000
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error with model '{model}': {e}")
        st.code(traceback.format_exc())
        return None

def parse_ai_output(text: str) -> list:
    clips = []
    sections = re.split(r'\*\*Short Title:\*\*', text.strip())
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        try:
            full_section_text = "**Short Title:**" + section
            title_match = re.search(r'\*\*Short Title:\*\*\s*(.*?)\n', full_section_text)
            title = title_match.group(1).strip() if title_match else f"Untitled Franken-Clip {i+1}"
            
            theme_match = re.search(r'\*\*Theme Category:\*\*\s*(.*?)\n', full_section_text)
            theme_category = theme_match.group(1).strip() if theme_match else "General"
            
            strategy_match = re.search(r'\*\*Viral Strategy:\*\*(.*?)(?=\n\*\*|$)', full_section_text, re.DOTALL)
            viral_strategy = strategy_match.group(1).strip() if strategy_match else "No strategy provided."
            
            approved_match = re.search(r'- APPROVED:\s*YES', full_section_text, re.IGNORECASE)
            if not approved_match:
                st.warning(f"Skipping clip '{title}' because it was not marked as 'APPROVED: YES'.")
                continue

            segments_match = re.search(r'\*\*Selected Segments:\*\*(.*?)\*\*Coherence Validation:\*\*', full_section_text, re.DOTALL)
            timestamps = []
            
            if segments_match:
                segments_text = segments_match.group(1)
                segment_pattern = r'SEGMENT\s+\d+:\s*([\d:,]+)\s*-->\s*([\d:,]+)\s*-\s*(.*?)\s*\[(HOOK|BUILD|PAYOFF|CONTEXT)\]'
                found_segments = re.findall(segment_pattern, segments_text, re.IGNORECASE)
                
                for seg_num, (start_str, end_str, text_content, purpose) in enumerate(found_segments, 1):
                    start_sec = parse_srt_timestamp(start_str)
                    end_sec = parse_srt_timestamp(end_str)
                    if end_sec > start_sec:
                        timestamps.append({
                            "segment_num": seg_num, "start_str": start_str.strip(), "end_str": end_str.strip(),
                            "start_sec": start_sec, "end_sec": end_sec, "duration": end_sec - start_sec,
                            "text": text_content.strip(), "label": purpose.strip().upper(),
                        })
            
            if timestamps and len(timestamps) >= 2:
                clips.append({
                    "title": title, "theme_category": theme_category, "num_segments": len(timestamps),
                    "viral_strategy": viral_strategy, "script": ' ... '.join([t['text'] for t in timestamps]),
                    "timestamps": timestamps
                })
                st.success(f"✅ AI proposed a clip: '{title}' with {len(timestamps)} precision segments.")
            else:
                st.warning(f"⚠️ Could not parse valid segments for '{title}'.")
                
        except Exception as e:
            st.warning(f"Could not parse a Franken-Clip section. Error: {e}")
            
    return clips

def download_drive_file(drive_url: str, download_path: str) -> str:
    """Downloads a Google Drive file and verifies its integrity."""
    try:
        output_path = os.path.join(download_path, 'downloaded_video.mp4')
        gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            raise Exception("Downloaded file is missing or empty.")
        with VideoFileClip(output_path) as clip:
            duration = clip.duration
        if duration is None or duration <= 0:
            raise Exception("Video file is corrupted.")
        st.info(f"Verified downloaded file. Duration: {duration:.2f}s.")
        return output_path
    except Exception as e:
        raise Exception(f"Google Drive download failed: {e}.")

def generate_clips_progressively(video_path: str, clips_data: list, output_dir: str):
    """Generator function that creates FRANKEN-CLIPS."""
    try:
        source_video = VideoFileClip(video_path)
        video_duration = source_video.duration
    except Exception as e:
        st.error(f"Fatal Error: Could not open the main video file. Error: {e}")
        return

    for i, clip_data in enumerate(clips_data):
        st.info(f"Processing Clip {i+1}/{len(clips_data)}: '{clip_data['title']}'")
        subclips = []
        valid_segments = []
        try:
            for ts in clip_data["timestamps"]:
                start_time, end_time = ts['start_sec'], ts['end_sec']
                if start_time < video_duration and end_time <= video_duration and start_time < end_time:
                    subclips.append(source_video.subclip(start_time, end_time))
                    valid_segments.append({
                        "label": ts['label'], "start": ts['start_str'], "end": ts['end_str'], "duration": ts['duration']
                    })
                    st.write(f"  ✅ Cutting `{ts['label']}` segment: `{ts['start_str']} -> {ts['end_str']}` ({ts['duration']:.1f}s)")
                else:
                    st.warning(f"  ⚠️ Skipping segment {ts['label']} as it's out of video bounds.")
            
            if not subclips:
                st.error(f"❌ No valid segments found for '{clip_data['title']}'. Skipping.")
                continue

            final_clip = concatenate_videoclips(subclips)
            total_duration = final_clip.duration
            safe_title = re.sub(r'[^\w\s-]', '', clip_data['title']).strip().replace(' ', '_')
            output_filepath = os.path.join(output_dir, f"clip_{i+1}_{safe_title[:30]}.mp4")
            
            st.info(f"🎥 Rendering: '{clip_data['title']}' ({total_duration:.1f}s)...")
            final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac", temp_audiofile=f'temp-audio-franken_{i}.m4a', remove_temp=True, logger='bar')
            
            yield {
                "path": output_filepath, "title": clip_data['title'], "theme_category": clip_data.get('theme_category', 'General'),
                "num_segments": len(valid_segments), "total_duration": total_duration,
                "viral_strategy": clip_data.get('viral_strategy', 'N/A'), "script": clip_data['script'],
                "valid_segments": valid_segments
            }
            st.success(f"✅ Generated: {os.path.basename(output_filepath)}")
        except Exception as e:
            st.error(f"❌ Failed to generate clip '{clip_data['title']}': {e}")
        finally:
            if 'final_clip' in locals(): final_clip.close()
            for sc in subclips: sc.close()
    
    source_video.close()

# ---
# 3. STREAMLIT APP
# ---

def main():
    st.set_page_config(page_title="Precision Franken-Clip Generator", layout="wide", page_icon="✂️")
    
    st.title("✂️ Precision Franken-Clip Generator")
    st.markdown("This tool reads your detailed transcript and finds the perfect phrases to stitch together a viral story.")

    if 'results' not in st.session_state:
        st.session_state.results = []

    # MODIFIED: Sidebar now includes AI provider and model selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("1. Inputs")
        video_url = st.text_input("Public Google Drive URL", placeholder="https://drive.google.com/...")
        uploaded_transcript = st.file_uploader("Upload Granular SRT Transcript", type=["srt", "txt"])
        st.info("For best results, use a **granular (word-level)** SRT file.")

        st.subheader("2. Generation Settings")
        clips_count = st.slider("Number of Clips to Generate:", 1, 5, 2)

        st.subheader("3. AI Settings")
        # For now, we only implement OpenAI, but this structure allows for future expansion
        ai_provider = st.selectbox("AI Provider:", ["OpenAI"]) #, "Google"
        
        selected_model = ""
        if ai_provider == "OpenAI":
            openai_api_key = get_openai_api_key()
            available_models = fetch_openai_models(openai_api_key)
            selected_model = st.selectbox(
                "Select OpenAI Model:",
                available_models,
                help="`gpt-4o` is powerful but slower. `gpt-4o-mini` is faster and cheaper."
            )
        
        st.markdown("---")
        with st.expander("ℹ️ How Precision Editing Works"):
            st.markdown("""
            - **1. Granular Input:** You provide a detailed transcript.
            - **2. AI Grouping:** The AI reads all the small lines and intelligently groups them into complete thoughts.
            - **3. Precision Cut:** It calculates the exact start/end time of that group and cuts only that segment.
            - **4. Narrative Stitch:** It repeats this to build a seamless, high-impact short.
            """)

    if st.button("🚀 Generate Precision Clips", type="primary", use_container_width=True):
        if not video_url or not uploaded_transcript:
            st.error("❌ Please provide both a video URL and a transcript file.")
            return
        if not selected_model:
            st.error("❌ Please select an AI model in the sidebar.")
            return

        st.session_state.results = []
        
        with st.status("🚀 Starting Precision Editing Process...", expanded=True) as status:
            try:
                status.update(label="Step 1/5: Loading granular transcript...")
                transcript_content = read_transcript_file(uploaded_transcript)
                if not transcript_content: raise ValueError("Transcript is empty or invalid.")
                
                status.update(label=f"Step 2/5: AI is analyzing transcript using {selected_model}...")
                # MODIFIED: Pass the selected model to the function
                ai_response = analyze_transcript_with_llm(transcript_content, clips_count, selected_model)
                if not ai_response: raise ValueError("AI analysis failed.")
                
                status.update(label="Step 3/5: Parsing AI's edit decisions...")
                clips_data = parse_ai_output(ai_response)
                if not clips_data:
                    st.error("❌ AI failed to propose valid clips.")
                    with st.expander("🔍 Show Raw AI Response for Debugging"):
                        st.text_area("AI Response", ai_response, height=300)
                    raise ValueError("Parsing failed.")
                
                persistent_dir = "generated_clips"
                if os.path.exists(persistent_dir): shutil.rmtree(persistent_dir)
                os.makedirs(persistent_dir)
                    
                with tempfile.TemporaryDirectory() as temp_dir:
                    status.update(label="Step 4/5: Downloading source video...")
                    video_path = download_drive_file(video_url, temp_dir)
                    
                    status.update(label="Step 5/5: Making the cuts and rendering videos...")
                    final_clips = [clip for clip in generate_clips_progressively(video_path, clips_data, persistent_dir)]
                    st.session_state.results = final_clips
                
                if not st.session_state.results:
                     status.update(label="Process finished, but no clips were generated.", state="error")
                else:
                    status.update(label=f"🎉 All {len(st.session_state.results)} clips are ready!", state="complete")

            except Exception as e:
                status.update(label=f"An error occurred: {e}", state="error")
                st.error(traceback.format_exc())

    if st.session_state.results:
        st.markdown("---")
        st.header("✅ Your Generated Precision Clips")
        
        for i, clip in enumerate(st.session_state.results):
            st.subheader(f"🎬 {i+1}. {clip['title']}")
            
            col_video, col_info = st.columns([3, 2])
            with col_video:
                if os.path.exists(clip['path']):
                    st.video(clip['path'])
                    with open(clip['path'], "rb") as file:
                        st.download_button(label="⬇️ Download Clip", data=file, file_name=os.path.basename(clip['path']), mime="video/mp4", key=f"dl_{i}")
                else:
                    st.error("File not found.")

            with col_info:
                st.metric("Theme", f"🎯 {clip.get('theme_category', 'N/A')}")
                st.metric("Total Duration", f"{clip.get('total_duration', 0):.1f}s")
                st.metric("Logical Segments", f"{clip.get('num_segments', 0)} parts")
            
            with st.expander("📊 Edit Breakdown & Viral Strategy"):
                st.markdown("**Viral Strategy:**")
                st.info(clip.get('viral_strategy', 'Not provided.'))
                st.markdown("**Precision Cuts:**")
                for seg in clip.get('valid_segments', []):
                    st.markdown(f"- **{seg['label']}**: `{seg['start']} → {seg['end']}` ({seg['duration']:.1f}s)")
                st.markdown("**Final Script:**")
                st.text_area("Script", clip.get('script', ''), height=100, key=f"script_{i}", disabled=True)

            st.markdown("---")

if __name__ == "__main__":
    main()
