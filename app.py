# Optimized Local Video Processor - Using Best Practices from Working Code
import os
import json
import re
import tempfile
import shutil
import streamlit as st
import traceback
import gc
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import crop, resize
import gdown
from openai import OpenAI

# ----------
# Configuration
# ----------

st.set_page_config(
    page_title="Optimized SRT ClipStitcher", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_api_key() -> str:
    """Get OpenAI API key from secrets."""
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    return os.getenv("OPENAI_API_KEY", "")

def cleanup_memory():
    """Force garbage collection to free memory."""
    gc.collect()

# ----------
# Smart Download with Verification (From Working Code)
# ----------

def extract_drive_file_id(drive_url: str) -> str:
    """Extract file ID from Google Drive URL patterns."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'open\?id=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    
    raise ValueError("Could not extract file ID from Google Drive URL")

def download_and_verify_video(drive_url: str, download_path: str) -> tuple:
    """
    Smart download with verification like the working code.
    Returns (success: bool, video_path: str, error_message: str)
    """
    try:
        output_path = os.path.join(download_path, 'source_video.mp4')
        
        st.info("📥 Downloading video from Google Drive...")
        
        # Use gdown with fuzzy matching (same as working code)
        gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
        
        # Verify file exists and has reasonable size
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            return False, None, "Downloaded file is missing or too small (likely corrupted)"
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        # Memory check - warn if file is very large
        if file_size_mb > 500:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f}MB). Processing may be slow.")
        else:
            st.info(f"✅ Downloaded {file_size_mb:.1f}MB")
        
        # Verify video integrity with MoviePy BEFORE processing
        try:
            with VideoFileClip(output_path) as test_clip:
                duration = test_clip.duration
                width, height = test_clip.size
                fps = test_clip.fps
                
            if duration is None or duration <= 0:
                return False, None, "Video file is corrupted (duration is zero or None)"
                
            st.success(f"✅ Video verified: {width}x{height} @ {fps}fps, {duration:.1f}s")
            return True, output_path, ""
            
        except Exception as e:
            return False, None, f"Video file appears corrupted: {e}"
            
    except Exception as e:
        return False, None, f"Download failed: {e}. Ensure the link is public and correct."

# ----------
# AI Analysis Functions
# ----------

def get_system_prompt() -> str:
    """System prompt for AI analysis."""
    return """
You are an expert viral video editor. Analyze the provided SRT transcript and select compelling segments for a viral short video.

REQUIREMENTS:
1. Select 2-5 separate segments that tell a cohesive story
2. Total duration: 15-45 seconds (optimized for memory)
3. Clear narrative arc: Hook -> Development -> Payoff
4. Each segment should be 3-15 seconds long

OUTPUT FORMAT (JSON only):
{
  "segments": [
    {
      "start": "00:00:04,439",
      "end": "00:00:06,169", 
      "text": "The biggest mistake..."
    }
  ],
  "reason": "Why this combination works",
  "title": "Catchy title"
}

CRITICAL: Use exact SRT timestamp format and keep total duration under 45 seconds.
"""

def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format to seconds."""
    try:
        time_parts = time_str.split(',')
        h, m, s = time_parts[0].split(':')
        ms = time_parts[1] if len(time_parts) > 1 else '0'
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    except Exception:
        return 0

def parse_srt(srt_content: str) -> tuple:
    """Parse SRT file content."""
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\n*$)', re.DOTALL)
    matches = pattern.findall(srt_content)
    
    segments = []
    transcript = []
    
    for match in matches:
        index, start, end, text = match
        text = text.strip().replace('\n', ' ')
        segments.append({"index": int(index), "start": start, "end": end, "text": text})
        transcript.append(f"[{index}] {start} --> {end} | {text}")
    
    return segments, "\n".join(transcript)

def analyze_transcript(transcript: str, client: OpenAI) -> dict:
    """Get AI analysis of transcript."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Analyze this transcript:\n\n{transcript}"}
            ],
            temperature=0.5,
            max_tokens=1500,  # Reduced for efficiency
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"AI analysis error: {e}")
        return {}

# ----------
# Progressive Video Processing (From Working Code)
# ----------

def generate_clips_progressively(video_path: str, segments: list, make_vertical: bool, output_dir: str):
    """
    Generator function that processes clips one by one with proper memory management.
    Yields completed clips as they're processed.
    """
    source_video = None
    
    try:
        st.info("🎬 Loading source video...")
        source_video = VideoFileClip(video_path)
        video_duration = source_video.duration
        width, height = source_video.size
        
        st.info(f"📊 Video loaded: {width}x{height}, {video_duration:.1f}s")
        
        for i, segment in enumerate(segments):
            clip_title = f"Clip {i+1}"
            st.info(f"🔄 Processing {clip_title}...")
            
            subclips = []
            final_clip = None
            
            try:
                start_time = srt_time_to_seconds(segment["start"])
                end_time = srt_time_to_seconds(segment["end"])
                
                # Validate timestamps
                if start_time >= end_time or start_time >= video_duration:
                    st.warning(f"⚠️ Invalid timestamp for {clip_title}. Skipping.")
                    continue
                
                end_time = min(end_time, video_duration)
                duration = end_time - start_time
                
                st.info(f"⏱️ Extracting {duration:.1f}s from {segment['start']} to {segment['end']}")
                
                # Create subclip
                subclip = source_video.subclip(start_time, end_time)
                
                # Apply vertical transformation if requested
                if make_vertical:
                    st.info("📱 Converting to vertical format...")
                    clip_w, clip_h = subclip.size
                    target_aspect = 9.0 / 16.0
                    current_aspect = float(clip_w) / clip_h
                    
                    if current_aspect > target_aspect:  # Too wide
                        new_width = int(clip_h * target_aspect)
                        subclip = crop(subclip, width=new_width, x_center=clip_w/2)
                    else:  # Too tall
                        new_height = int(clip_w / target_aspect)
                        subclip = crop(subclip, height=new_height, y_center=clip_h/2)
                    
                    # Resize to standard vertical resolution
                    subclip = resize(subclip, height=min(1080, subclip.h))
                
                # Generate safe filename
                safe_title = re.sub(r'[^\w\s-]', '', segment.get("text", f"clip_{i+1}"))[:20]
                safe_title = safe_title.strip().replace(' ', '_')
                output_filename = f"clip_{i+1}_{safe_title}.mp4"
                output_filepath = os.path.join(output_dir, output_filename)
                
                # Render clip with optimized settings
                st.info("🎥 Rendering clip...")
                subclip.write_videofile(
                    output_filepath,
                    codec="libx264",
                    audio_codec="aac",
                    threads=1,  # Single thread to reduce memory
                    preset='ultrafast',  # Faster encoding
                    verbose=False,
                    logger=None,
                    temp_audiofile=f'temp_audio_{i}.m4a',
                    remove_temp=True
                )
                
                # Verify output file
                if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 1024:
                    clip_size_mb = os.path.getsize(output_filepath) / (1024 * 1024)
                    
                    yield {
                        "path": output_filepath,
                        "title": segment.get("text", clip_title)[:50] + "...",
                        "duration": duration,
                        "size_mb": clip_size_mb,
                        "success": True
                    }
                    
                    st.success(f"✅ {clip_title} completed ({clip_size_mb:.1f}MB)")
                else:
                    st.error(f"❌ {clip_title} failed to generate")
                    
            except Exception as e:
                st.error(f"❌ Error processing {clip_title}: {e}")
                yield {
                    "path": None,
                    "title": clip_title,
                    "error": str(e),
                    "success": False
                }
            finally:
                # Clean up this iteration's objects
                if 'subclip' in locals() and subclip:
                    subclip.close()
                cleanup_memory()
                
    except Exception as e:
        st.error(f"❌ Error loading source video: {e}")
    finally:
        # Clean up source video
        if source_video:
            source_video.close()
        cleanup_memory()

# ----------
# Main Streamlit App
# ----------

def main():
    st.title("🎬 Optimized SRT ClipStitcher")
    st.markdown("**Smart video processing with progressive loading and memory optimization**")
    
    # Memory usage info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Memory Usage", "Optimized")
    with col2:
        st.metric("Processing", "Progressive")
    with col3:
        st.metric("Max File Size", "1GB+")
    
    # Initialize session state
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            "video_path": None,
            "srt_content": None,
            "analysis_result": None,
            "generated_clips": [],
            "processing_complete": False
        }
    
    state = st.session_state.app_state
    
    # API key check
    api_key = get_api_key()
    if not api_key:
        st.error("❌ OpenAI API key required. Add to Streamlit secrets.")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    
    # Sidebar inputs
    st.sidebar.header("📥 Configuration")
    
    drive_url = st.sidebar.text_input(
        "🔗 Google Drive Video URL",
        placeholder="https://drive.google.com/file/d/...",
        help="Make sure the video is shared publicly"
    )
    
    srt_file = st.sidebar.file_uploader("📄 SRT Transcript", type=["srt"])
    
    make_vertical = st.sidebar.checkbox("📱 Create Vertical Clips (9:16)", value=True)
    
    # Memory management
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Memory Management")
    if st.sidebar.button("🗑️ Force Cleanup"):
        cleanup_memory()
        st.sidebar.success("Memory cleaned!")
    
    # Process inputs
    if drive_url and not state["video_path"]:
        with st.spinner("📥 Downloading and verifying video..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                success, video_path, error = download_and_verify_video(drive_url, temp_dir)
                
                if success:
                    # Move to persistent location
                    persistent_dir = "temp_videos"
                    os.makedirs(persistent_dir, exist_ok=True)
                    
                    # Clean up old files
                    for f in os.listdir(persistent_dir):
                        try:
                            os.remove(os.path.join(persistent_dir, f))
                        except:
                            pass
                    
                    persistent_path = os.path.join(persistent_dir, "source_video.mp4")
                    shutil.move(video_path, persistent_path)
                    state["video_path"] = persistent_path
                    
                else:
                    st.error(f"❌ {error}")
                    st.stop()
    
    if srt_file and not state["srt_content"]:
        try:
            state["srt_content"] = srt_file.getvalue().decode("utf-8")
            st.success("✅ SRT transcript loaded")
        except Exception as e:
            st.error(f"❌ Error reading SRT: {e}")
    
    # Main interface
    if not (state["video_path"] and state["srt_content"]):
        st.info("👆 Please provide both Google Drive URL and SRT file to continue")
        st.stop()
    
    # Show video
    if state["video_path"] and os.path.exists(state["video_path"]):
        file_size = os.path.getsize(state["video_path"]) / (1024 * 1024)
        st.caption(f"Source video: {file_size:.1f}MB")
        st.video(state["video_path"])
    
    # Show transcript
    with st.expander("📄 View Transcript"):
        st.text_area("SRT Content", state["srt_content"], height=200)
    
    # AI Analysis
    if not state["analysis_result"]:
        if st.button("🤖 Analyze Transcript", type="primary"):
            with st.spinner("🧠 AI analyzing transcript..."):
                try:
                    _, transcript_text = parse_srt(state["srt_content"])
                    analysis = analyze_transcript(transcript_text, client)
                    
                    if analysis and "segments" in analysis:
                        state["analysis_result"] = analysis
                        st.rerun()
                    else:
                        st.error("❌ Analysis failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {e}")
    else:
        # Show analysis results
        analysis = state["analysis_result"]
        
        st.subheader("🎯 AI Analysis Results")
        st.success(f"**Title:** {analysis.get('title', 'Untitled')}")
        st.info(f"**Strategy:** {analysis.get('reason', 'No reason provided')}")
        
        segments = analysis.get("segments", [])
        total_duration = sum(
            srt_time_to_seconds(s["end"]) - srt_time_to_seconds(s["start"]) 
            for s in segments
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Selected Segments", f"{len(segments)} clips")
        with col2:
            st.metric("Total Duration", f"{total_duration:.1f}s")
        
        # Show segments
        with st.expander("📝 Selected Segments"):
            for i, seg in enumerate(segments, 1):
                duration = srt_time_to_seconds(seg["end"]) - srt_time_to_seconds(seg["start"])
                st.markdown(f"**{i}.** `{seg['start']} → {seg['end']}` ({duration:.1f}s)")
                st.write(f"*\"{seg['text']}\"*")
        
        # Process clips
        if not state["processing_complete"]:
            if st.button("🚀 Generate Clips", type="primary"):
                st.markdown("---")
                st.header("🎬 Generating Clips...")
                
                # Create output directory
                output_dir = "generated_clips"
                os.makedirs(output_dir, exist_ok=True)
                
                # Clean old clips
                for f in os.listdir(output_dir):
                    try:
                        os.remove(os.path.join(output_dir, f))
                    except:
                        pass
                
                # Progressive processing
                generated_clips = []
                clip_generator = generate_clips_progressively(
                    state["video_path"], 
                    segments, 
                    make_vertical, 
                    output_dir
                )
                
                # Process and display clips as they complete
                for clip_result in clip_generator:
                    if clip_result["success"]:
                        generated_clips.append(clip_result)
                        
                        # Display the just-completed clip
                        st.subheader(f"✅ {clip_result['title']}")
                        col_video, col_info = st.columns([2, 1])
                        
                        with col_video:
                            st.video(clip_result['path'])
                            
                            with open(clip_result['path'], "rb") as file:
                                st.download_button(
                                    "⬇️ Download",
                                    file,
                                    file_name=os.path.basename(clip_result['path']),
                                    key=f"download_{len(generated_clips)}"
                                )
                        
                        with col_info:
                            st.metric("Duration", f"{clip_result['duration']:.1f}s")
                            st.metric("Size", f"{clip_result['size_mb']:.1f}MB")
                        
                        st.markdown("---")
                
                state["generated_clips"] = generated_clips
                state["processing_complete"] = True
                st.success(f"🎉 Generated {len(generated_clips)} clips successfully!")
        
        else:
            # Show existing clips
            st.header("✅ Your Generated Clips")
            for i, clip in enumerate(state["generated_clips"]):
                if os.path.exists(clip['path']):
                    st.subheader(f"🎬 {clip['title']}")
                    col_video, col_info = st.columns([2, 1])
                    
                    with col_video:
                        st.video(clip['path'])
                        with open(clip['path'], "rb") as file:
                            st.download_button(
                                "⬇️ Download",
                                file,
                                file_name=os.path.basename(clip['path']),
                                key=f"existing_download_{i}"
                            )
                    
                    with col_info:
                        st.metric("Duration", f"{clip['duration']:.1f}s")
                        st.metric("Size", f"{clip['size_mb']:.1f}MB")
                    
                    st.markdown("---")
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over"):
        # Clean up files
        for key, value in state.items():
            if '_path' in key and value and isinstance(value, str) and os.path.exists(value):
                try:
                    os.remove(value)
                except:
                    pass
        
        # Clean up directories
        for dir_name in ["temp_videos", "generated_clips"]:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name, ignore_errors=True)
        
        # Reset state
        st.session_state.app_state = {
            "video_path": None,
            "srt_content": None,
            "analysis_result": None,
            "generated_clips": [],
            "processing_complete": False
        }
        cleanup_memory()
        st.rerun()

if __name__ == "__main__":
    main()
