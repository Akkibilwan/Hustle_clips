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

Create FRANKEN-CLIPS by stitching together NON-CONTIGUOUS segments from different parts of the transcript to create viral 30-60 second shorts.

**IMPORTANT: The transcript has been pre-processed. Each segment contains complete thoughts with proper durations.**

---

🔥 PRIORITIZED THEMES:
- 💰 **Money & Career** — salaries, struggle, financial risks
- 💥 **Vulnerability** — fear, failure, doubt  
- 🎯 **Transformations** — loss to clarity, confusion to purpose
- 🎭 **Industry Secrets** — what people don't see behind success
- 💡 **Advice** — wisdom, frameworks
- 🧨 **Breaking Norms** — challenging stereotypes

---

🎯 STRATEGY:
- Select 3-4 segments from DIFFERENT parts (2+ minutes apart)  
- Each segment 5-20 seconds long
- Create: HOOK → BUILD → PAYOFF structure
- Total duration: 25-50 seconds

---

📦 OUTPUT FORMAT:

**Short Title:** [Viral title with emoji]
**Theme Category:** [Money/Vulnerability/Transformation/Industry/Advice/Norms]  
**Number of Segments:** [3-4]

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

def merge_consecutive_srt_lines(srt_content: str) -> str:
    """
    Intelligently merge consecutive SRT lines into longer, coherent segments.
    Takes start time of first line and end time of last line in each group.
    """
    # Parse individual SRT lines
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n*$)', re.DOTALL)
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
        
        # End group if we hit punctuation that indicates sentence end
        if text.endswith(('.', '!', '?')):
            should_end_group = True
        
        # End group if current group is getting long (8+ seconds or 15+ words)
        if len(current_group) >= 15:
            should_end_group = True
            
        # End group if there's a pause in the next timestamp (gap > 1 second)
        if i < len(matches) - 1:
            current_end_sec = parse_srt_timestamp(end)
            next_start_sec = parse_srt_timestamp(matches[i+1][1])
            if next_start_sec - current_end_sec > 1.0:
                should_end_group = True
        
        # End group if we're at the last item
        if i == len(matches) - 1:
            should_end_group = True
            
        # Create merged segment
        if should_end_group and current_group:
            first_start = current_group[0][1]
            last_end = current_group[-1][2]
            merged_text = ' '.join(current_text)
            
            # Only include if the merged text is substantial (3+ seconds and 5+ words)
            duration = parse_srt_timestamp(last_end) - parse_srt_timestamp(first_start)
            word_count = len(merged_text.split())
            
            if duration >= 3.0 and word_count >= 5:
                merged_segments.append({
                    'index': group_counter,
                    'start': first_start,
                    'end': last_end,
                    'text': merged_text,
                    'duration': duration,
                    'word_count': word_count
                })
                group_counter += 1
            
            # Reset for next group
            current_group = []
            current_text = []
    
    # Convert back to SRT format
    merged_srt = ""
    for seg in merged_segments:
        merged_srt += f"{seg['index']}\n{seg['start']} --> {seg['end']}\n{seg['text']}\n\n"
    
    st.info(f"📝 Merged {len(matches)} individual lines into {len(merged_segments)} coherent segments")
    return merged_srt

def validate_clip_coherence(title: str, segments: list, script_preview: str) -> tuple:
    """
    Validate if the selected segments create a coherent, viral-worthy clip.
    Returns (is_valid: bool, reason: str)
    """
    
    # Check 1: Minimum duration requirements
    total_duration = sum(seg.get('duration', 0) for seg in segments)
    if total_duration < 20:
        return False, f"Total duration {total_duration:.1f}s too short (need 20+ seconds)"
    
    if total_duration > 70:
        return False, f"Total duration {total_duration:.1f}s too long (max 70 seconds)"
    
    # Check 2: Individual segment length
    for i, seg in enumerate(segments, 1):
        duration = seg.get('duration', 0)
        if duration < 3:
            return False, f"Segment {i} too short ({duration:.1f}s) - need 3+ seconds"
        if duration > 25:
            return False, f"Segment {i} too long ({duration:.1f}s) - max 25 seconds"
    
    # Check 3: Time gaps between segments (should be spread out)
    if len(segments) >= 2:
        for i in range(len(segments) - 1):
            current_end = parse_srt_timestamp(segments[i]['end_str'])
            next_start = parse_srt_timestamp(segments[i+1]['start_str'])
            gap = next_start - current_end
            
            if gap < 30:  # Less than 30 seconds apart
                return False, f"Segments {i+1} and {i+2} too close together ({gap:.1f}s gap) - need 30+ seconds"
    
    # Check 4: Content quality - look for filler words or incomplete thoughts
    script_lower = script_preview.lower()
    filler_ratio = sum(script_lower.count(word) for word in ['um', 'uh', 'like', 'you know']) / len(script_lower.split())
    
    if filler_ratio > 0.15:  # More than 15% filler words
        return False, f"Too many filler words ({filler_ratio:.1%}) - not engaging enough"
    
    # Check 5: Has question/answer or setup/payoff structure
    has_question = any(seg.get('text', '').strip().endswith('?') for seg in segments)
    has_strong_words = any(word in script_lower for word in [
        'secret', 'mistake', 'truth', 'reality', 'never', 'always', 'biggest', 'worst', 'best'
    ])
    
    if not (has_question or has_strong_words):
        return False, "No compelling hook or strong language detected"
    
    return True, "✅ Clip passes all coherence checks"

def read_transcript_file(uploaded_file) -> str:
    try:
        content = uploaded_file.read().decode("utf-8")
        
        # Check if it's word-level SRT (many short segments)
        lines = content.strip().split('\n')
        srt_blocks = content.split('\n\n')
        
        if len(srt_blocks) > 50:  # Likely word-level SRT
            st.info("🔍 Detected word-level SRT. Merging consecutive lines into coherent segments...")
            return merge_consecutive_srt_lines(content)
        else:
            st.info("📄 Standard SRT format detected.")
            return content
            
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
            
            # Extract number of segments
            segments_match = re.search(r'\*\*Number of Segments:\*\*\s*(.*?)(?:\n|\*\*)', section)
            num_segments = segments_match.group(1).strip() if segments_match else "3"

            # Extract coherence validation
            coherence_match = re.search(r'\*\*Coherence Validation:\*\*(.*?)(?=\*\*Viral Strategy:\*\*)', section, re.DOTALL)
            coherence_validation = coherence_match.group(1).strip() if coherence_match else ""

            # Extract viral strategy
            strategy_match = re.search(r'\*\*Viral Strategy:\*\*(.*?)(?:\n\*\*|$)', section, re.DOTALL)
            viral_strategy = strategy_match.group(1).strip() if strategy_match else "No strategy provided."

            # Look for coherence validation with multiple possible formats
            coherence_patterns = [
                r'\*\*Coherence Validation:\*\*(.*?)(?=\*\*Viral Strategy:\*\*)',
                r'\*\*Coherence Check:\*\*(.*?)(?=\*\*Viral Strategy:\*\*)',
                r'\*\*Validation:\*\*(.*?)(?=\*\*Viral Strategy:\*\*)'
            ]
            
            coherence_validation = ""
            for pattern in coherence_patterns:
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    coherence_validation = match.group(1).strip()
                    break
            
            if not coherence_validation:
                coherence_validation = "No validation found"
                st.info(f"📝 No coherence validation found for '{title}' - will attempt to parse anyway")

            # Extract viral strategy
            strategy_match = re.search(r'\*\*Viral Strategy:\*\*(.*?)(?:\n\*\*|$)', section, re.DOTALL)
            viral_strategy = strategy_match.group(1).strip() if strategy_match else "No strategy provided."

            # Extract selected segments with flexible parsing
            segments_patterns = [
                r'\*\*Selected Segments:\*\*(.*?)(?=\*\*Coherence)',
                r'\*\*Timestamp Segments:\*\*(.*?)(?=\*\*Script)',
                r'\*\*Segments:\*\*(.*?)(?=\*\*Coherence)'
            ]
            
            segments_text = ""
            for pattern in segments_patterns:
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    segments_text = match.group(1).strip()
                    break
            
            timestamps = []
            
            if segments_text:
                # Parse segments with multiple possible formats
                segment_lines = segments_text.strip().split('\n')
                
                for line in segment_lines:
                    line = line.strip()
                    if line and ('SEGMENT' in line or ':' in line):
                        # Try multiple parsing patterns
                        patterns = [
                            r'SEGMENT\s+(\d+):\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-\s*(.*?)\s*\[(.*?)\]',
                            r'SEGMENT\s+(\d+):\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*(.*)',
                            r'(\d+)[:\.]?\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*(.*)'
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, line)
                            if match:
                                groups = match.groups()
                                if len(groups) >= 3:
                                    segment_num = groups[0]
                                    start_str = groups[1].replace('.', ',')
                                    end_str = groups[2].replace('.', ',')
                                    text = groups[3] if len(groups) > 3 else f"Segment {segment_num}"
                                    purpose = groups[4] if len(groups) > 4 else "Content"
                                    
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
                                    break

            # Accept clips with any valid timestamps (relaxed validation)
            if timestamps:
                st.info(f"📍 Found {len(timestamps)} valid segments for '{title}'")
                
                clips.append({
                    "title": title,
                    "type": "Franken-Clip",
                    "theme_category": theme_category,
                    "num_segments": len(timestamps),
                    "coherence_validation": coherence_validation,
                    "viral_strategy": viral_strategy,
                    "script": ' '.join([t.get('text', '') for t in timestamps]),
                    "timestamps": timestamps,
                    "validation_passed": True,
                    "validation_reason": f"Found {len(timestamps)} segments"
                })
                st.success(f"✅ Parsed '{title}' with {len(timestamps)} segments")
            else:
                st.warning(f"⚠️ No valid segments found for '{title}'")
                    
        except Exception as e:
            st.warning(f"Could not parse Franken-Clip section {i}: {e}")
    
    return clips

def parse_ai_output_relaxed(text: str) -> list:
    """
    Relaxed parsing that accepts clips even if they don't pass all checks.
    Used as fallback when strict parsing fails.
    """
    clips = []
    sections = re.split(r'\*\*Short Title:\*\*', text)
    
    st.info(f"🔍 Found {len(sections)-1} potential clips to parse")
    
    for i, section in enumerate(sections[1:], 1):
        try:
            title_match = re.search(r'^(.*?)(?:\n|\*\*)', section, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else f"Franken-Clip {i}"
            
            # Extract theme category
            theme_match = re.search(r'\*\*Theme Category:\*\*\s*(.*?)(?:\n|\*\*)', section)
            theme_category = theme_match.group(1).strip() if theme_match else "General"
            
            # Look for ANY timestamp pattern
            timestamp_patterns = [
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',  # Standard format
                r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})',  # Dot format
            ]
            
            timestamps = []
            for pattern in timestamp_patterns:
                matches = re.findall(pattern, section)
                for j, (start_str, end_str) in enumerate(matches):
                    start_str = start_str.replace('.', ',')  # Normalize to comma format
                    end_str = end_str.replace('.', ',')
                    
                    start_sec = parse_srt_timestamp(start_str)
                    end_sec = parse_srt_timestamp(end_str)
                    duration = end_sec - start_sec
                    
                    if duration > 1:  # At least 1 second
                        timestamps.append({
                            "segment_num": j + 1,
                            "start_str": start_str,
                            "end_str": end_str,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration": duration,
                            "text": f"Segment {j+1} text",
                            "label": f"Segment {j+1}",
                            "purpose": "Content"
                        })
            
            if timestamps:
                st.info(f"📍 Found {len(timestamps)} timestamps for '{title}'")
                
                clips.append({
                    "title": title,
                    "type": "Franken-Clip",
                    "theme_category": theme_category,
                    "num_segments": len(timestamps),
                    "coherence_validation": "Relaxed parsing - validation bypassed",
                    "viral_strategy": "Generated from relaxed parsing",
                    "script": f"Script for {title}",
                    "timestamps": timestamps,
                    "validation_passed": True,
                    "validation_reason": "Relaxed validation - timestamps found"
                })
                st.success(f"✅ Added '{title}' via relaxed parsing")
            else:
                st.warning(f"⚠️ No valid timestamps found for '{title}'")
                
        except Exception as e:
            st.warning(f"❌ Error in relaxed parsing for section {i}: {e}")
    
    return clips
                    
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
                
                # Debug: Show raw AI response
                with st.expander("🔍 Debug: Raw AI Response"):
                    st.text_area("AI Response", ai_response, height=400)
                
                # Try relaxed parsing as fallback
                st.info("🔄 Attempting relaxed parsing...")
                clips_data = parse_ai_output_relaxed(ai_response)
                
                if not clips_data:
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
