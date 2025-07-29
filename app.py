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
# 1. SYSTEM PROMPT - Simplified and Clear
# ---
SYSTEM_PROMPT = """
You are an expert YouTube Shorts strategist specializing in FRANKEN-CLIPS.

Create FRANKEN-CLIPS by stitching together NON-CONTIGUOUS segments from different parts of the transcript to create viral 30-60 second shorts.

**IMPORTANT: The transcript has been pre-processed. Each segment contains complete thoughts with proper durations.**

---

🔥 PRIORITIZED THEMES:
- 💰 **Money & Career** — salaries, struggle, financial risks
- 💥 **Vulnerability** — fear, failure, doubt  
- 🎯 **Transformations** — loss to clarity, confusion to purpose
- 🎭 **Industry Secrets** — what people don\'t see behind success
- 💡 **Advice** — wisdom, frameworks
- 🧨 **Breaking Norms** — challenging stereotypes

---

🎯 STRATEGY:
- Select 3-5 segments from DIFFERENT parts (2+ minutes apart)  
- Each segment 10-30 seconds long
- Create: HOOK → BUILD → PAYOFF structure
- Total duration: 30-60 seconds

---

📦 OUTPUT FORMAT:

**Short Title:** [Viral title with emoji]
**Theme Category:** [Money/Vulnerability/Transformation/Industry/Advice/Norms]  
**Number of Segments:** [3-5]

**Selected Segments:**
SEGMENT 1: 00:01:23,450 --> 00:01:35,200 - Hook content here [HOOK]
SEGMENT 2: 00:05:15,300 --> 00:05:28,800 - Build content here [BUILD]  
SEGMENT 3: 00:12:03,100 --> 00:12:18,900 - Payoff content here [PAYOFF]

**Coherence Validation:**
- Strong hook: YES - creates curiosity
- Logical flow: YES - builds tension to payoff  
- Satisfying payoff: YES - delivers insight
- APPROVED: YES

**Viral Strategy:**
[Why this combination works and targets the theme]

---

🛑 REQUIREMENTS:
- Use EXACT timestamps from transcript
- Select segments that are 2+ minutes apart
- Ensure APPROVED: YES in validation
- Focus on prioritized themes
- Create compelling narrative arc

Generate 2-3 Franken-Clips following this format exactly.
"""

# ---
# 2. HELPER FUNCTIONS
# ---

def get_openai_api_key() -> str:
    return st.secrets.get("openai", {}).get("api_key", "")

def parse_srt_timestamp(timestamp_str: str) -> float:
    """Convert SRT timestamp format to total seconds."""
    timestamp_str = timestamp_str.strip().replace(\\',\\', \'.\\')
    try:
        time_parts = timestamp_str.split(\\':\\')
        if len(time_parts) == 3:
            h, m, s_ms = time_parts
            return int(h) * 3600 + int(m) * 60 + float(s_ms)
        elif len(time_parts) == 2:
            m, s_ms = time_parts
            return int(m) * 60 + float(s_ms)
        return float(time_parts[0])
    except Exception:
        return 0.0

def merge_consecutive_srt_lines(srt_content: str) -> str:
    """
    Intelligently merge consecutive SRT lines into longer, coherent segments.
    Takes start time of first line and end time of last line in each group.
    """
    # Parse individual SRT lines
    pattern = re.compile(r\'(\\d+)\\n(\\d{2}:\\d{2}:\\d{2},\\d{3}) --> (\\d{2}:\\d{2}:\\d{2},\\d{3})\\n(.*?)(?=\\n\\n|\\n*$)\\', re.DOTALL)
    matches = pattern.findall(srt_content)
    
    if not matches:
        return srt_content
    
    merged_segments = []
    current_group = []
    current_text = []
    group_counter = 1
    
    for i, (index, start, end, text) in enumerate(matches):
        text = text.strip()
        
        # Skip empty lines
        if not text:
            continue
            
        current_group.append((index, start, end, text))
        current_text.append(text)
        
        # Check if we should end this group and start a new one
        should_end_group = False
        
        # End group if we hit punctuation that indicates sentence end AND the group is already reasonably long
        if text.endswith((\\'.\\', \'!\\', \'?\\')) and len(current_text) >= 5:
            should_end_group = True
        
        # End group if current group is getting long (increased from 15 to 30 individual lines)
        if len(current_group) >= 30:
            should_end_group = True
            
        # End group if there\'s a pause in the next timestamp (gap > 1.5 seconds - increased from 1.0)
        if i < len(matches) - 1:
            current_end_sec = parse_srt_timestamp(end)
            next_start_sec = parse_srt_timestamp(matches[i+1][1])
            if next_start_sec - current_end_sec > 1.5:
                should_end_group = True
        
        # End group if we\'re at the last item
        if i == len(matches) - 1:
            should_end_group = True
            
        # Create merged segment
        if should_end_group and current_group:
            first_start = current_group[0][1]
            last_end = current_group[-1][2]
            merged_text = \' \'.join(current_text)
            
            # Only include if the merged text is substantial (increased from 3+ seconds to 5+ seconds and 10+ words)
            duration = parse_srt_timestamp(last_end) - parse_srt_timestamp(first_start)
            word_count = len(merged_text.split())
            
            if duration >= 5.0 and word_count >= 10:
                merged_segments.append({
                    \'index\': group_counter,
                    \'start\': first_start,
                    \'end\': last_end,
                    \'text\': merged_text,
                    \'duration\': duration,
                    \'word_count\': word_count
                })
                group_counter += 1
            
            # Reset for next group
            current_group = []
            current_text = []
    
    # Convert back to SRT format
    merged_srt = ""
    for seg in merged_segments:
        merged_srt += f"{seg[\'index\']}\\n{seg[\'start\']} --> {seg[\'end\']}\\n{seg[\'text\]}\\n\\n"
    
    st.info(f"📝 Merged {len(matches)} individual lines into {len(merged_segments)} coherent segments")
    return merged_srt

def read_transcript_file(uploaded_file) -> str:
    try:
        content = uploaded_file.read().decode("utf-8")
        
        # Check if it\'s word-level SRT (many short segments)
        lines = content.strip().split(\'\\n\')
        srt_blocks = content.split(\'\\n\\n\')
        
        if len(srt_blocks) > 50:  # Likely word-level SRT
            st.info("🔍 Detected word-level SRT. Merging consecutive lines into coherent segments...")
            return merge_consecutive_srt_lines(content)
        else:
            st.info("📄 Standard SRT format detected.")
            return content
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def analyze_transcript_with_llm(transcript: str, count: int):
    user_content = f"{transcript}\\n\\nPlease generate {count} unique Franken-Clips following the exact format specified."
    
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
    sections = re.split(r\'\\*\\*Short Title:\\*\\*\\', text)
    
    for i, section in enumerate(sections[1:], 1):
        try:
            title_match = re.search(r\'^(.*?)(?:\\n|\\*\\*)\\', section, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else f"Untitled Franken-Clip {i}"
            
            # Extract theme category
            theme_match = re.search(r\'\\*\\*Theme Category:\\*\\*\\s*(.*?)(?:\\n|\\*\\*)\\', section)
            theme_category = theme_match.group(1).strip() if theme_match else "General"
            
            # Extract viral strategy
            strategy_match = re.search(r\'\\*\\*Viral Strategy:\\*\\*([\s\S]*?)(?:\\n\\*\\*|$)\\', section, re.DOTALL)
            viral_strategy = strategy_match.group(1).strip() if strategy_match else "No strategy provided."

            # Extract selected segments
            segments_text_match = re.search(r\'\\*\\*Selected Segments:\\*\\*([\s\S]*?)(?=\\*\\*Coherence Validation:\\*\\*)\\', section, re.DOTALL)
            timestamps = []
            
            if segments_text_match:
                segments_text = segments_text_match.group(1)
                segment_lines = segments_text.strip().split(\'\\n\')
                
                for line in segment_lines:
                    line = line.strip()
                    if line.startswith(\'SEGMENT\'):
                        # Parse segment line
                        segment_match = re.search(r\'SEGMENT\\s+(\\d+):\\s*(\\d{2}:\\d{2}:\\d{2},\\d{3})\\s*-->\\s*(\\d{2}:\\d{2}:\\d{2},\\d{3})\\s*-\\s*(.*?)\\s*\\[(.*?)\\]\\', line)
                        if segment_match:
                            segment_num, start_str, end_str, text, purpose = segment_match.groups()
                            start_sec = parse_srt_timestamp(start_str)
                            end_sec = parse_srt_timestamp(end_str)
                            duration = end_sec - start_sec
                            
                            timestamps.append({
                                "segment_num": int(segment_num),
                                "start_str": start_str,
                                "end_str": end_str,
                                "start_sec": start_sec,
                                "end_sec": end_sec,
                                "duration": duration,
                                "text": text.strip(),
                                "label": purpose.strip(),
                                "purpose": purpose.strip()
                            })

            if timestamps:
                clips.append({
                    "title": title,
                    "type": "Franken-Clip",
                    "theme_category": theme_category,
                    "num_segments": len(timestamps),
                    "viral_strategy": viral_strategy,
                    "script": \' \'.join([t.get(\'text\', \'\') for t in timestamps]),
                    "timestamps": timestamps
                })
                st.success(f"✅ Parsed \'{title}\' with {len(timestamps)} segments")
            else:
                st.warning(f"⚠️ No valid segments found for \'{title}\'")
                
        except Exception as e:
            st.warning(f"Could not parse Franken-Clip section {i}: {e}")
    
    return clips

def download_drive_file(drive_url: str, download_path: str) -> str:
    """Downloads a Google Drive file and verifies its integrity."""
    try:
        output_path = os.path.join(download_path, \'downloaded_video.mp4\')
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
        st.info(f"Processing Franken-Clip {i+1}/{len(clips_data)}: \'{clip_data[\'title\]}\'")
        st.info(f"📊 Stitching {clip_data[\'num_segments\]} segments together...")
        
        try:
            subclips = []
            valid_segments = []
            
            # Process each timestamp segment
            for j, ts in enumerate(clip_data["timestamps"]):
                start_time, end_time = ts[\'start_sec\'], ts[\'end_sec\']
                segment_duration = end_time - start_time
                segment_label = ts.get(\'label\', f"Segment {ts.get(\'segment_num\', j+1)}")
                
                if start_time < video_duration and end_time <= video_duration:
                    subclip = source_video.subclip(start_time, end_time)
                    subclips.append(subclip)
                    valid_segments.append({
                        "segment_num": ts.get(\'segment_num\', j + 1),
                        "start": ts[\'start_str\'],
                        "end": ts[\'end_str\'],
                        "duration": segment_duration,
                        "label": segment_label
                    })
                    st.info(f"  ✅ {segment_label}: {ts[\'start_str\']} → {ts[\'end_str\]} ({segment_duration:.1f}s)")
                else:
                    st.warning(f"  ⚠️ {segment_label}: {ts[\'start_str\']} → {ts[\'end_str\]} is out of bounds. Skipping.")
            
            if not subclips:
                st.error(f"❌ No valid segments for Franken-Clip \'{clip_data[\'title\]}\'. Skipping.")
                continue
            
            if len(subclips) < 3:
                st.warning(f"⚠️ Only {len(subclips)} valid segments found. Franken-Clips work better with 3+ segments.")

            # Calculate total duration
            total_duration = sum(seg["duration"] for seg in valid_segments)
            st.info(f"🎬 Total Franken-Clip duration: {total_duration:.1f} seconds")

            # Concatenate all segments in order
            final_clip = concatenate_videoclips(subclips)
            
            safe_title = re.sub(r\'[^\\w\\s-]\', \'\', clip_data[\'title\]).strip().replace(\' \', \'_\')
            output_filepath = os.path.join(output_dir, f"franken_clip_{i+1}_{safe_title[:20]}.mp4")
            
            st.info("🎥 Rendering Franken-Clip...")
            final_clip.write_videofile(
                output_filepath, 
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile=f\'temp-audio-franken_{i}.m4a\', 
                remove_temp=True, 
                logger=None
            )
            
            # YIELD the completed Franken-Clip with all segment details
            yield {
                "path": output_filepath,
                "title": clip_data[\'title\'],
                "type": "Franken-Clip",
                "theme_category": clip_data.get(\'theme_category\', \'General\'),
                "num_segments": len(valid_segments),
                "total_duration": total_duration,
                "viral_strategy": clip_data.get(\'viral_strategy\', \'No strategy provided\'),
                "script": clip_data[\'script\'],
                "timestamps": clip_data[\'timestamps\'],
                "valid_segments": valid_segments
            }
            st.success(f"✅ Generated Franken-Clip: {clip_data[\'title\]}")

        except Exception as e:
            st.error(f"❌ Failed to generate Franken-Clip \'{clip_data[\'title\]}\': {e}")
        finally:
            if \'final_clip\' in locals(): 
                final_clip.close()
            if \'subclips\' in locals():
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
        **Franken-Clips** are viral short videos created by stitching together **non-contiguous segments** from different parts of 
        your transcript to create highly engaging and shareable content. They are designed to capture attention quickly 
        and deliver a compelling narrative or insight.
        
        **Key Characteristics:**
        - **Non-Contiguous:** Segments are pulled from different, often distant, parts of the original video.
        - **Narrative Arc:** Each Franken-Clip aims to tell a mini-story with a Hook, Build, and Payoff.
        - **Thematic Focus:** Clips are centered around specific viral themes (e.g., Money, Vulnerability, Transformation).
        - **Short & Punchy:** Optimized for platforms like YouTube Shorts, TikTok, and Instagram Reels.
        """
    
    st.subheader("Upload Transcript (SRT) and Video")
    
    uploaded_srt_file = st.file_uploader("Upload SRT File", type=["srt"])
    drive_link = st.text_input("Google Drive Link to Video (ensure public access)", "")
    
    num_clips_to_generate = st.slider("Number of Franken-Clips to Generate", min_value=1, max_value=5, value=2)

    if uploaded_srt_file and drive_link:
        if st.button("Generate Franken-Clips"):
            with tempfile.TemporaryDirectory() as temp_dir:
                srt_content = read_transcript_file(uploaded_srt_file)
                
                if srt_content:
                    st.info("Transcript processed. Analyzing with AI...")
                    
                    try:
                        ai_response = analyze_transcript_with_llm(srt_content, num_clips_to_generate)
                        
                        if ai_response:
                            st.subheader("AI Analysis Complete. Parsing Clips...")
                            clips_data = parse_ai_output(ai_response)
                            
                            if clips_data:
                                st.subheader("Downloading Video...")
                                try:
                                    video_path = download_drive_file(drive_link, temp_dir)
                                    st.success(f"Video downloaded to: {video_path}")
                                    
                                    st.subheader("Generating Franken-Clips...")
                                    output_clips_dir = os.path.join(temp_dir, "output_clips")
                                    os.makedirs(output_clips_dir, exist_ok=True)
                                    
                                    generated_clips_info = []
                                    for clip_info in generate_clips_progressively(video_path, clips_data, output_clips_dir):
                                        generated_clips_info.append(clip_info)
                                        
                                    if generated_clips_info:
                                        st.subheader("Generated Clips:")
                                        for clip in generated_clips_info:
                                            st.markdown(f"### {clip[\'title\]}")
                                            st.write(f"**Theme:** {clip[\'theme_category\]}")
                                            st.write(f"**Total Duration:** {clip[\'total_duration\]:.1f} seconds")
                                            st.write(f"**Segments:** {clip[\'num_segments\]}")
                                            st.write(f"**Viral Strategy:** {clip[\'viral_strategy\]}")
                                            st.write("**Script:**")
                                            st.info(clip[\'script\])
                                            
                                            st.download_button(
                                                label=f"Download {clip[\'title\]}.mp4",
                                                data=open(clip[\'path\'], "rb").read(),
                                                file_name=os.path.basename(clip[\'path\']),
                                                mime="video/mp4"
                                            )
                                            st.video(clip[\'path\'])
                                    else:
                                        st.warning("No Franken-Clips were generated.")
                                        
                                except Exception as e:
                                    st.error(f"Video processing error: {e}")
                            else:
                                st.warning("AI did not return any valid clip data.")
                        else:
                            st.error("AI analysis failed or returned no content.")
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
                else:
                    st.error("Failed to read or process SRT content.")
    
    st.markdown("""
    ---
    Built with ❤️ by Your Name/Team
    """)

if __name__ == "__main__":
    main()


