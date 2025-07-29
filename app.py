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
You are an expert YouTube Shorts strategist and narrative video editor. Your specialty is crafting **high-retention, viral YouTube Shorts by stitching together non-contiguous moments** from long-form interviews or podcasts. These are called **Franken-Clips**.

🎯 OBJECTIVE:  
From the provided transcript, generate **only Franken-Clips** (no direct continuous segments). Your goal is to create short-form micro-stories using **timestamps from different parts of the interview** that, when combined, sound like a seamless, engaging, and emotionally resonant video.

These Shorts must feel **natural, complete, and coherent when played aloud**, even though they are stitched. The final clip must sound like a conversation, not a mash-up. Do not include filler, vague lines, or disjointed transitions.

---

🧠 DEEP READING FIRST:
Before generating anything, you **must read and deeply understand the entire transcript**.
- Do not extract clips based on keywords alone.
- Use emotional and narrative intelligence to identify meaningful hooks, emotional climaxes, and memorable takeaways.
- Every clip you suggest must tell a **clear, engaging mini-story**, even if watched in isolation.

---

🎬 HOW TO BUILD A FRANKEN-CLIP (ONLY RETURN IF ALL THESE ARE TRUE):
1. **Hook (0–3s)**: A strong, attention-grabbing moment — shocking stat, raw confession, stereotype-breaker, or direct quote.
2. **Context (optional)**: Include only if a brief setup is required for clarity.
3. **Insight (middle)**: The core message — an “aha” realization, lesson, or emotional turning point.
4. **Takeaway (end)**: A final quote or reflection that’s easy to remember, share, or feel moved by.

💡 The entire stitched clip must **make sense when heard aloud** — avoid clips that feel jarring, context-less, or overly abstract.

---

🔥 PRIORITIZED THEMES:
You should prioritize clips that explore:
- 💰 **Money & Career Truths** — salaries, struggle, financial risks.
- 💥 **Vulnerability** — fear, failure, loneliness, doubt.
- 🎯 **Transformations** — from loss to clarity, confusion to purpose.
- 🎭 **Industry Realities** — what people don’t see behind success.
- 💡 **Advice** — powerful, lived wisdom or personal frameworks.
- 🧨 **Stereotype-Busting** — breaking societal norms or stigma.

---

📦 OUTPUT FORMAT (repeat for each Franken-Clip):

**Short Title:** [Catchy title with emoji]
**Estimated Total Duration:** [e.g., 42 seconds]
**Type:** Franken-Clip
**Number of Segments:** [e.g., 5 segments]

**Timestamp Segments:**
SEGMENT 1: 00:01:23,450 --> 00:01:27,200
SEGMENT 2: 00:05:15,300 --> 00:05:19,800
SEGMENT 3: 00:12:03,100 --> 00:12:08,900
SEGMENT 4: 00:18:45,200 --> 00:18:50,500
SEGMENT 5: 00:25:12,800 --> 00:25:18,300

**Script:**
[Exact dialogue from each segment in order - no modifications]

**Script:**  
[Exact dialogue from each segment in order. Dialogue must flow like natural speech. Do not rewrite or combine non-matching tones.]

**Rationale for Virality:**  
[Explain why this clip works — emotionally powerful arc, high relatability, truth bomb, surprising advice, payoff quote, etc.]

---

⚠️ FINAL RULES:
- ✅ Do NOT include direct clips (contiguous timestamps).
- ✅ Do NOT fabricate dialogue or paraphrase — use exact lines.
- ✅ DO stitch only if the narrative sounds natural when spoken.
- ✅ DO discard any idea that doesn’t follow the full viral arc.
- ✅ DO prefer emotionally clear or surprising Franken-Clips over superficial or vague ones.

---

ONLY return Franken-Clips that are **story-worthy, emotionally resonant, and coherent when played aloud**. If no good Franken-Clip can be made, return nothing.


Generate ONLY Franken-Clips following this exact format.
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
            
            # Extract number of segments
            segments_match = re.search(r'\*\*Number of Segments:\*\*\s*(.*?)(?:\n|\*\*)', section)
            num_segments = segments_match.group(1).strip() if segments_match else "Unknown"

            rationale_match = re.search(r'\*\*Rationale:\*\*(.*?)(?:\n\*\*|$)', section, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

            script_match = re.search(r'\*\*Script:\*\*(.*?)(?=\*\*Rationale:\*\*)', section, re.DOTALL)
            script = script_match.group(1).strip() if script_match else "Script not found."

            # Extract ALL individual timestamp segments
            timestamp_text_match = re.search(r'\*\*Timestamp Segments:\*\*(.*?)(?=\*\*Script:\*\*)', section, re.DOTALL)
            timestamps = []
            if timestamp_text_match:
                timestamp_text = timestamp_text_match.group(1)
                # Find all SEGMENT patterns
                segment_matches = re.findall(r'SEGMENT\s+\d+:\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', timestamp_text)
                for start_str, end_str in segment_matches:
                    start_sec = parse_srt_timestamp(start_str)
                    end_sec = parse_srt_timestamp(end_str)
                    timestamps.append({
                        "start_str": start_str, 
                        "end_str": end_str, 
                        "start_sec": start_sec, 
                        "end_sec": end_sec
                    })

            if timestamps and len(timestamps) >= 3:  # Must have at least 3 segments for Franken-Clips
                clips.append({
                    "title": title, 
                    "type": "Franken-Clip", 
                    "num_segments": len(timestamps),
                    "rationale": rationale,
                    "script": script, 
                    "timestamps": timestamps
                })
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
            
            # Process each timestamp segment
            for j, ts in enumerate(clip_data["timestamps"]):
                start_time, end_time = ts['start_sec'], ts['end_sec']
                segment_duration = end_time - start_time
                
                if start_time < video_duration and end_time <= video_duration:
                    subclip = source_video.subclip(start_time, end_time)
                    subclips.append(subclip)
                    valid_segments.append({
                        "segment_num": j + 1,
                        "start": ts['start_str'],
                        "end": ts['end_str'],
                        "duration": segment_duration
                    })
                    st.info(f"  ✅ Segment {j+1}: {ts['start_str']} → {ts['end_str']} ({segment_duration:.1f}s)")
                else:
                    st.warning(f"  ⚠️ Segment {j+1}: {ts['start_str']} → {ts['end_str']} is out of bounds. Skipping.")
            
            if not subclips:
                st.error(f"❌ No valid segments for Franken-Clip '{clip_data['title']}'. Skipping.")
                continue
            
            if len(subclips) < 3:
                st.warning(f"⚠️ Only {len(subclips)} valid segments found. Franken-Clips need at least 3 segments.")

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
                "num_segments": len(valid_segments),
                "total_duration": total_duration,
                "rationale": clip_data['rationale'],
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
            with st.expander(f"📱 Franken-Clip {i}: {clip['title']}"):
                st.markdown(f"**Segments:** {clip['num_segments']} parts")
                st.markdown("**Timestamp Segments:**")
                for j, ts in enumerate(clip['timestamps'], 1):
                    duration = parse_srt_timestamp(ts['end_str']) - parse_srt_timestamp(ts['start_str'])
                    st.markdown(f"  {j}. `{ts['start_str']} → {ts['end_str']}` ({duration:.1f}s)")
                st.markdown(f"**Strategy:** {clip['rationale']}")
        
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
                        st.metric("Segments Used", f"{clip['num_segments']} parts")
                        st.metric("Total Duration", f"{clip['total_duration']:.1f}s")
                    
                    # Detailed segment breakdown
                    st.markdown("### 📊 Segment Breakdown")
                    segment_cols = st.columns(min(len(clip['valid_segments']), 3))
                    for i, segment in enumerate(clip['valid_segments']):
                        col_idx = i % len(segment_cols)
                        with segment_cols[col_idx]:
                            st.markdown(f"**Segment {segment['segment_num']}**")
                            st.code(f"{segment['start']} → {segment['end']}")
                            st.caption(f"{segment['duration']:.1f} seconds")
                    
                    # Additional details in expanders
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("💡 Strategy & Rationale"):
                            st.info(clip['rationale'])
                    with col2:
                        with st.expander("📜 Full Script"):
                            st.text_area("", clip['script'], height=100, key=f"script_{new_path}")
                    
                    # Complete timestamp listing
                    with st.expander("⏰ All Timestamp Segments Used"):
                        for i, ts in enumerate(clip['timestamps'], 1):
                            duration = parse_srt_timestamp(ts['end_str']) - parse_srt_timestamp(ts['start_str'])
                            st.markdown(f"**Segment {i}:** `{ts['start_str']} → {ts['end_str']}` ({duration:.1f}s)")
                    
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
                    st.metric("Segments", f"{clip.get('num_segments', 'Unknown')} parts")
                    if 'total_duration' in clip:
                        st.metric("Duration", f"{clip['total_duration']:.1f}s")
                
                # Segment breakdown
                if 'valid_segments' in clip:
                    st.markdown("### 📊 Segment Breakdown")
                    segment_cols = st.columns(min(len(clip['valid_segments']), 3))
                    for i, segment in enumerate(clip['valid_segments']):
                        col_idx = i % len(segment_cols)
                        with segment_cols[col_idx]:
                            st.markdown(f"**Segment {segment['segment_num']}**")
                            st.code(f"{segment['start']} → {segment['end']}")
                            st.caption(f"{segment['duration']:.1f} seconds")
                
                # Details in expanders
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("💡 Strategy & Rationale"):
                        st.info(clip['rationale'])
                with col2:
                    with st.expander("📜 Full Script"):
                        st.text_area("", clip['script'], height=100, key=f"script_replay_{clip['path']}")
                
                # Complete timestamp listing
                with st.expander("⏰ All Timestamp Segments Used"):
                    for i, ts in enumerate(clip['timestamps'], 1):
                        duration = parse_srt_timestamp(ts['end_str']) - parse_srt_timestamp(ts['start_str'])
                        st.markdown(f"**Segment {i}:** `{ts['start_str']} → {ts['end_str']}` ({duration:.1f}s)")
                
                st.markdown("---")

if __name__ == "__main__":
    main()
