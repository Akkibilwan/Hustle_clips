# Franken-Clip Generator - Extract Non-Contiguous Viral Segments
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
# 1. SYSTEM PROMPT - Franken-Clips Only
# ---
SYSTEM_PROMPT = """
You are an expert YouTube Shorts strategist specializing in FRANKEN-CLIPS.

Your ONLY job is to create FRANKEN-CLIPS by stitching together NON-CONTIGUOUS segments from different parts of the transcript to create viral 30-60 second shorts.

---

🔥 PRIORITIZED THEMES:
You should prioritize clips that explore:
- 💰 **Money & Career Truths** — salaries, struggle, financial risks.
- 💥 **Vulnerability** — fear, failure, loneliness, doubt.
- 🎯 **Transformations** — from loss to clarity, confusion to purpose.
- 🎭 **Industry Realities** — what people don't see behind success.
- 💡 **Advice** — powerful, lived wisdom or personal frameworks.
- 🧨 **Stereotype-Busting** — breaking societal norms or stigma.

---

🎯 FRANKEN-CLIP STRATEGY (for WORD-LEVEL SRT with precise timestamps):

**CRITICAL: COMBINE MULTIPLE SRT LINES FOR LONGER SEGMENTS**
- Since each SRT line is 1-2 words, you MUST combine 10-30 consecutive lines to create meaningful 8-15 second segments
- Each segment should contain COMPLETE THOUGHTS or SENTENCES
- Don't break in the middle of sentences or ideas

**SEGMENT LENGTH REQUIREMENTS:**
- **HOOK SEGMENT**: 8-15 seconds (combine 20-40 SRT lines for a complete hook)
- **CONTEXT SEGMENT**: 6-12 seconds (combine 15-30 SRT lines for background)
- **BUILD SEGMENT**: 8-15 seconds (combine 20-40 SRT lines for tension/stakes)
- **PAYOFF SEGMENT**: 10-18 seconds (combine 25-45 SRT lines for the revelation/wisdom)
- **TOTAL DURATION**: 35-55 seconds combined

**NARRATIVE STRUCTURE (MANDATORY):**
1. **HOOK** (8-15s): Compelling question, surprising statement, or cliffhanger from early in transcript
2. **CONTEXT** (6-12s): Brief background or setup from a different timestamp  
3. **BUILD** (8-15s): Tension, stakes, or emotional escalation from another timestamp
4. **PAYOFF** (10-18s): The big reveal, wisdom, or emotional resolution from much later

---

🛑 STRICT REQUIREMENTS:
- ONLY create Franken-Clips (NO direct continuous clips)
- Use EXACT transcript dialogue - no modifications whatsoever
- **COMBINE 10-30 consecutive SRT lines** to create each segment
- Each segment must be **8-18 seconds long** (not 2-4 seconds!)
- Each segment must contain **COMPLETE THOUGHTS**
- Segments must be from DIFFERENT parts of the video (at least 2-3 minutes apart)
- **READ THE COMBINED TEXT** - ensure it makes sense as a coherent segment
- Focus on the PRIORITIZED THEMES above

---

📦 OUTPUT FORMAT (repeat for each Franken-Clip):

**Short Title:** [Viral title with emoji focusing on prioritized themes]
**Theme Category:** [Which prioritized theme: Money/Vulnerability/Transformation/Industry/Advice/Stereotype-Busting]
**Estimated Total Duration:** [e.g., 42 seconds]
**Type:** Franken-Clip
**Number of Segments:** [Always 4 segments: HOOK + CONTEXT + BUILD + PAYOFF]

**Timestamp Segments:**
SEGMENT 1: 00:01:23,450 --> 00:01:37,200 [HOOK - compelling question/statement]
SEGMENT 2: 00:05:15,300 --> 00:05:26,800 [CONTEXT - background/setup]  
SEGMENT 3: 00:12:03,100 --> 00:12:18,900 [BUILD - tension/stakes/emotion]
SEGMENT 4: 00:18:45,200 --> 00:19:02,500 [PAYOFF - revelation/wisdom/resolution]

**Script Preview:**
[Show the COMBINED text from each segment to verify it makes sense - include 3-4 lines from each segment to demonstrate coherence]

**Coherence Check:**
[Verify that each segment contains complete thoughts and the narrative flow makes sense. State "APPROVED" only if all segments are coherent and create a compelling story arc]

**Viral Strategy:**
[Explain why this specific combination creates maximum retention and targets the chosen theme]

---

🎯 EXAMPLES OF PROPER SEGMENT COMBINATION:

**WRONG** (Too short, incomplete thought):
SEGMENT 1: 00:01:23,450 --> 00:01:25,200 [Text: "The biggest"]

**RIGHT** (Complete thought, proper length):  
SEGMENT 1: 00:01:23,450 --> 00:01:37,200 [Text: "The biggest mistake I made in my career was thinking that being nice would get me promoted. I spent three years being the person who said yes to everything, who stayed late, who never pushed back. And you know what happened? I got passed over for promotion twice."]

---

🛑 CRITICAL REQUIREMENTS:
- Each Franken-Clip must have EXACTLY 4 segments following HOOK → CONTEXT → BUILD → PAYOFF structure
- Each segment must be 8-18 seconds long (combine multiple SRT lines!)
- **COHERENCE CHECK MANDATORY** - only approve clips where all segments make complete sense
- Segments must jump across significant time gaps (2+ minutes apart)
- Focus on viral themes for maximum engagement

Generate ONLY Franken-Clips that pass the coherence check and follow the proper structure.
"""

# ---
# 2. HELPER FUNCTIONS
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
    user_content = f"{transcript}\n\nPlease generate {count} unique Franken-Clips following the exact format specified."
    
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
            title = title_match.group(1).strip() if title_match else f"Untitled Franken-Clip {i}"
            
            # Extract theme category
            theme_match = re.search(r'\*\*Theme Category:\*\*\s*(.*?)(?:\n|\*\*)', section)
            theme_category = theme_match.group(1).strip() if theme_match else "General"
            
            # Extract number of segments (should always be 4 now)
            segments_match = re.search(r'\*\*Number of Segments:\*\*\s*(.*?)(?:\n|\*\*)', section)
            num_segments = segments_match.group(1).strip() if segments_match else "4"

            # Extract coherence check
            coherence_match = re.search(r'\*\*Coherence Check:\*\*(.*?)(?:\n\*\*)', section, re.DOTALL)
            coherence_check = coherence_match.group(1).strip() if coherence_match else ""

            # Extract viral strategy
            strategy_match = re.search(r'\*\*Viral Strategy:\*\*(.*?)(?:\n\*\*|$)', section, re.DOTALL)
            viral_strategy = strategy_match.group(1).strip() if strategy_match else "No strategy provided."

            # Extract script preview
            script_match = re.search(r'\*\*Script Preview:\*\*(.*?)(?=\*\*Coherence Check:\*\*)', section, re.DOTALL)
            script = script_match.group(1).strip() if script_match else "Script not found."

            # Extract ALL individual timestamp segments with enhanced labels
            timestamp_text_match = re.search(r'\*\*Timestamp Segments:\*\*(.*?)(?=\*\*Script Preview:\*\*)', section, re.DOTALL)
            timestamps = []
            if timestamp_text_match:
                timestamp_text = timestamp_text_match.group(1)
                # Find all SEGMENT patterns with structure labels
                segment_matches = re.findall(r'SEGMENT\s+(\d+):\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\[(.*?)\]', timestamp_text)
                for segment_num, start_str, end_str, label in segment_matches:
                    start_sec = parse_srt_timestamp(start_str)
                    end_sec = parse_srt_timestamp(end_str)
                    duration = end_sec - start_sec
                    
                    # Validate segment length (8-18 seconds as per new requirements)
                    if duration < 6:
                        st.warning(f"⚠️ Segment {segment_num} is only {duration:.1f}s - may be too short for coherent content")
                    elif duration > 20:
                        st.warning(f"⚠️ Segment {segment_num} is {duration:.1f}s - may be too long for retention")
                    
                    timestamps.append({
                        "segment_num": int(segment_num),
                        "start_str": start_str, 
                        "end_str": end_str, 
                        "start_sec": start_sec, 
                        "end_sec": end_sec,
                        "duration": duration,
                        "label": label.strip() if label else f"Segment {segment_num}"
                    })

            # Only accept clips that pass coherence check and have proper structure
            if timestamps and len(timestamps) == 4 and "APPROVED" in coherence_check.upper():
                clips.append({
                    "title": title, 
                    "type": "Franken-Clip", 
                    "theme_category": theme_category,
                    "num_segments": len(timestamps),
                    "coherence_check": coherence_check,
                    "viral_strategy": viral_strategy,
                    "script": script, 
                    "timestamps": timestamps
                })
            elif "APPROVED" not in coherence_check.upper():
                st.warning(f"⚠️ Skipping Franken-Clip '{title}' - failed coherence check")
            elif len(timestamps) != 4:
                st.warning(f"⚠️ Skipping Franken-Clip '{title}' - needs exactly 4 segments (HOOK→CONTEXT→BUILD→PAYOFF)")
                
        except Exception as e:
            st.warning(f"Could not parse Franken-Clip section {i}: {e}")
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
    Generator function that creates FRANKEN-CLIPS by stitching multiple timestamp segments.
    """
    source_video = VideoFileClip(video_path)
    video_duration = source_video.duration
    
    for i, clip_data in enumerate(clips_data):
        st.info(f"Processing Franken-Clip {i+1}/{len(clips_data)}: '{clip_data['title']}'")
        st.info(f"📊 Stitching {clip_data['num_segments']} segments together...")
        
        try:
            subclips = []
            valid_segments = []
            
            # Process each timestamp segment with enhanced details
            for j, ts in enumerate(clip_data["timestamps"]):
                start_time, end_time = ts['start_sec'], ts['end_sec']
                segment_duration = end_time - start_time
                segment_label = ts.get('label', f"Segment {ts.get('segment_num', j+1)}")
                
                if start_time < video_duration and end_time <= video_duration:
                    subclip = source_video.subclip(start_time, end_time)
                    subclips.append(subclip)
                    valid_segments.append({
                        "segment_num": ts.get('segment_num', j + 1),
                        "start": ts['start_str'],
                        "end": ts['end_str'],
                        "duration": segment_duration,
                        "label": segment_label
                    })
                    st.info(f"  ✅ {segment_label}: {ts['start_str']} → {ts['end_str']} ({segment_duration:.1f}s)")
                else:
                    st.warning(f"  ⚠️ {segment_label}: {ts['start_str']} → {ts['end_str']} is out of bounds. Skipping.")
            
            if not subclips:
                st.error(f"❌ No valid segments for Franken-Clip '{clip_data['title']}'. Skipping.")
                continue
            
            if len(subclips) < 4:
                st.warning(f"⚠️ Only {len(subclips)} valid segments found. Enhanced Franken-Clips require exactly 4 segments (HOOK→CONTEXT→BUILD→PAYOFF).")
                continue

            # Calculate total duration
            total_duration = sum(seg["duration"] for seg in valid_segments)
            st.info(f"🎬 Total Franken-Clip duration: {total_duration:.1f} seconds")

            # Concatenate all segments in order
            final_clip = concatenate_videoclips(subclips)
            
            safe_title = re.sub(r'[^\w\s-]', '', clip_data['title']).strip().replace(' ', '_')
            output_filepath = os.path.join(output_dir, f"franken_clip_{i+1}_{safe_title[:20]}.mp4")
            
            st.info("🎥 Rendering Franken-Clip...")
            final_clip.write_videofile(
                output_filepath, 
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile=f'temp-audio-franken_{i}.m4a', 
                remove_temp=True, 
                logger=None
            )
            
            # YIELD the completed Franken-Clip with all segment details
            yield {
                "path": output_filepath,
                "title": clip_data['title'],
                "type": "Franken-Clip",
                "theme_category": clip_data.get('theme_category', 'General'),
                "num_segments": len(valid_segments),
                "total_duration": total_duration,
                "coherence_check": clip_data.get('coherence_check', 'Not checked'),
                "viral_strategy": clip_data.get('viral_strategy', 'No strategy provided'),
                "script": clip_data['script'],
                "timestamps": clip_data['timestamps'],  # Original timestamps
                "valid_segments": valid_segments  # Processed segments with details
            }
            st.success(f"✅ Generated Franken-Clip: {clip_data['title']}")

        except Exception as e:
            st.error(f"❌ Failed to generate Franken-Clip '{clip_data['title']}': {e}")
        finally:
            if 'final_clip' in locals(): 
                final_clip.close()
            if 'subclips' in locals():
                for sc in subclips: 
                    sc.close()

    source_video.close()

# ---
# 3. STREAMLIT APP
# ---

def main():
    st.set_page_config(page_title="Franken-Clip Generator", layout="wide", page_icon="🧩")
    
    st.title("🎬 Franken-Clip Generator")
    st.markdown("**Create viral Franken-Clips by stitching together non-contiguous segments from your videos.**")
    
    # Info about Franken-Clips
    with st.expander("ℹ️ What are Franken-Clips?"):
        st.markdown("""
        **Franken-Clips** are viral short videos created by stitching together **non-contiguous segments** from different parts of a long-form video.
        
        **How it works:**
        - Take a **hook** from one timestamp (question, setup, cliffhanger)
        - Jump to a different timestamp for the **payoff** (answer, reveal, punchline)
        - Skip boring middle parts, filler words, or weak segments
        - Create maximum retention by jumping to only the best moments
        
        **Example:** Hook from 2:15 → Jump to 15:30 → Jump to 28:45 → End at 35:20
        """)

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        video_url = st.text_input("Google Drive URL", placeholder="Paste your Google Drive URL here...")
        uploaded_transcript = st.file_uploader("Upload SRT/Transcript File", type=["srt", "txt"])
        clips_count = st.slider("Number of Franken-Clips to Generate:", 1, 5, 2)
        
        st.markdown("---")
        st.subheader("📋 Franken-Clip Requirements")
        st.info("• Minimum 3 segments per clip\n• Each segment 3-15 seconds\n• Total duration 30-60 seconds\n• Segments from different video parts")
    
    # Main action button
    if st.button("🚀 Generate Franken-Clips", type="primary", use_container_width=True):
        st.session_state.results = None # Clear previous results
        if not video_url or not uploaded_transcript:
            st.error("❌ Please provide both a video URL and a transcript file.")
            return

        with st.spinner("📖 Reading transcript..."):
            transcript_content = read_transcript_file(uploaded_transcript)
            if not transcript_content: return
        st.success("✅ Transcript loaded.")

        with st.spinner("🧠 Analyzing transcript for Franken-Clip opportunities..."):
            ai_response = analyze_transcript_with_llm(transcript_content, clips_count)
            if not ai_response: return
        st.success("✅ Franken-Clip analysis complete.")

        with st.spinner("📝 Parsing Franken-Clip recommendations..."):
            clips_data = parse_ai_output(ai_response)
            if not clips_data:
                st.error("❌ Could not parse any valid Franken-Clips from the AI response.")
                return
        st.success(f"✅ Parsed {len(clips_data)} Franken-Clip recommendations.")
        
        # Show analysis results and proceed directly to processing
        st.markdown("---")
        st.header("🎯 Franken-Clip Analysis Results")
        for i, clip in enumerate(clips_data, 1):
            with st.expander(f"📱 Franken-Clip {i}: {clip['title']} | 🎯 {clip.get('theme_category', 'General')}"):
                st.markdown(f"**Theme:** {clip.get('theme_category', 'General')}")
                st.markdown(f"**Segments:** {clip['num_segments']} parts")
                st.markdown("**Timestamp Segments:**")
                for ts in clip['timestamps']:
                    duration = parse_srt_timestamp(ts['end_str']) - parse_srt_timestamp(ts['start_str'])
                    label = ts.get('label', f"Segment {ts.get('segment_num', 'X')}")
                    st.markdown(f"  • `{ts['start_str']} → {ts['end_str']}` ({duration:.1f}s) - *{label}*")
                st.markdown(f"**Strategy:** {clip.get('viral_strategy', clip.get('rationale', 'No strategy provided'))}")
        
        # Proceed directly to generation without asking permission
        st.markdown("---")
        st.header("🎬 Generating Franken-Clips...")
        
        # --- Process the Franken-Clips ---
        st.session_state.results = {"type": "generator", "data": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with st.spinner("⬇️ Downloading video from Google Drive..."):
                    video_path = download_drive_file(video_url, temp_dir)
                st.success("✅ Video downloaded and verified.")

                persistent_dir = "generated_clips"
                if not os.path.exists(persistent_dir):
                    os.makedirs(persistent_dir)
                for f in os.listdir(persistent_dir):
                    os.remove(os.path.join(persistent_dir, f))
                
                # --- Progressive Franken-Clip Generation ---
                final_clips = []
                clip_generator = generate_clips_progressively(video_path, clips_data, temp_dir)
                for clip in clip_generator:
                    # Move file and display immediately
                    new_path = os.path.join(persistent_dir, os.path.basename(clip['path']))
                    shutil.move(clip['path'], new_path)
                    clip['path'] = new_path
                    final_clips.append(clip)
                    
                    # Display the Franken-Clip that was just generated
                    st.subheader(f"🎬 {clip['title']}")
                    
                    # Main video display
                    col_video, col_info = st.columns([3, 2])
                    with col_video:
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as file:
                            st.download_button("⬇️ Download Franken-Clip", file, file_name=os.path.basename(clip['path']), key=f"dl_{new_path}")
                    
                    with col_info:
                        st.metric("Type", "🧩 Franken-Clip")
                        st.metric("Theme", f"🎯 {clip.get('theme_category', 'General')}")
                        st.metric("Segments Used", f"{clip['num_segments']} parts")
                        st.metric("Total Duration", f"{clip['total_duration']:.1f}s")
                    
                    # Enhanced segment breakdown with labels
                    st.markdown("### 📊 Segment Breakdown")
                    segment_cols = st.columns(min(len(clip['valid_segments']), 3))
                    for i, segment in enumerate(clip['valid_segments']):
                        col_idx = i % len(segment_cols)
                        with segment_cols[col_idx]:
                            st.markdown(f"**{segment.get('label', f'Segment {segment['segment_num']}')}**")
                            st.code(f"{segment['start']} → {segment['end']}")
                            st.caption(f"{segment['duration']:.1f} seconds")
                    
                    # Additional details in expanders
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("🎯 Viral Strategy"):
                            st.info(clip.get('viral_strategy', 'No strategy provided'))
                    with col2:
                        with st.expander("📜 Full Script"):
                            st.text_area("", clip['script'], height=100, key=f"script_{new_path}")
                    
                    # Complete timestamp listing with labels
                    with st.expander("⏰ All Timestamp Segments Used"):
                        for ts in clip['timestamps']:
                            duration = parse_srt_timestamp(ts['end_str']) - parse_srt_timestamp(ts['start_str'])
                            label = ts.get('label', f"Segment {ts.get('segment_num', 'X')}")
                            st.markdown(f"**{label}:** `{ts['start_str']} → {ts['end_str']}` ({duration:.1f}s)")
                    
                    st.markdown("---")

                st.session_state.results["data"] = final_clips # Save final list
                st.success(f"🎉 All {len(final_clips)} Franken-Clips generated successfully!")

            except Exception as e:
                st.error(f"An error occurred during Franken-Clip generation: {e}")
                st.code(traceback.format_exc())

    # Display results from session state if they exist (for reruns)
    elif st.session_state.results:
        results = st.session_state.results
        st.markdown("---")
        
        if results["type"] == "generator":
            st.header("✅ Your Generated Franken-Clips")
            if not results["data"]:
                st.warning("No Franken-Clips were successfully generated.")
            
            for clip in results["data"]:
                st.subheader(f"🎬 {clip['title']}")
                
                # Main video and info
                col_video, col_info = st.columns([3, 2])
                with col_video:
                    if os.path.exists(clip['path']):
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as file:
                            st.download_button("⬇️ Download Franken-Clip", file, file_name=os.path.basename(clip['path']), key=f"dl_{clip['path']}")
                    else:
                        st.error("Franken-Clip file not found.")
                
                with col_info:
                    st.metric("Type", "🧩 Franken-Clip")
                    st.metric("Theme", f"🎯 {clip.get('theme_category', 'General')}")
                    st.metric("Segments", f"{clip.get('num_segments', 'Unknown')} parts")
                    if 'total_duration' in clip:
                        st.metric("Duration", f"{clip['total_duration']:.1f}s")
                
                # Enhanced segment breakdown with labels
                if 'valid_segments' in clip:
                    st.markdown("### 📊 Segment Breakdown")
                    segment_cols = st.columns(min(len(clip['valid_segments']), 3))
                    for i, segment in enumerate(clip['valid_segments']):
                        col_idx = i % len(segment_cols)
                        with segment_cols[col_idx]:
                            st.markdown(f"**{segment.get('label', f'Segment {segment['segment_num']}')}**")
                            st.code(f"{segment['start']} → {segment['end']}")
                            st.caption(f"{segment['duration']:.1f} seconds")
                
                # Details in expanders
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("🎯 Viral Strategy"):
                        st.info(clip.get('viral_strategy', clip.get('rationale', 'No strategy provided')))
                with col2:
                    with st.expander("📜 Full Script"):
                        st.text_area("", clip['script'], height=100, key=f"script_replay_{clip['path']}")
                
                # Complete timestamp listing with labels
                with st.expander("⏰ All Timestamp Segments Used"):
                    for ts in clip['timestamps']:
                        duration = parse_srt_timestamp(ts['end_str']) - parse_srt_timestamp(ts['start_str'])
                        label = ts.get('label', f"Segment {ts.get('segment_num', 'X')}")
                        st.markdown(f"**{label}:** `{ts['start_str']} → {ts['end_str']}` ({duration:.1f}s)")
                
                st.markdown("---")

if __name__ == "__main__":
    main()
