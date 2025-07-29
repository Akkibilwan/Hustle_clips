# Cloud-based Video Processing - No Local Downloads Required
import os
import json
import re
import streamlit as st
import requests
from openai import OpenAI
import time
from urllib.parse import parse_qs, urlparse

# ----------
# Cloud Video Processing Functions
# ----------

def extract_drive_file_id(drive_url: str) -> str:
    """Extract file ID from various Google Drive URL formats."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'open\?id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    
    raise ValueError("Could not extract file ID from Google Drive URL")

def get_direct_video_url(file_id: str) -> str:
    """Get direct streaming URL for Google Drive video."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def get_video_metadata_from_url(video_url: str) -> dict:
    """Get video metadata without downloading the file."""
    try:
        response = requests.head(video_url, allow_redirects=True)
        content_length = response.headers.get('content-length')
        content_type = response.headers.get('content-type', '')
        
        return {
            "url": video_url,
            "size_mb": int(content_length) / (1024 * 1024) if content_length else 0,
            "content_type": content_type,
            "streamable": "video" in content_type.lower()
        }
    except Exception as e:
        st.error(f"Error getting video metadata: {e}")
        return None

def create_video_processing_job(video_url: str, segments: list, make_vertical: bool = True) -> dict:
    """Create a video processing job using external API."""
    
    # This would integrate with services like:
    # - FFmpeg.wasm for browser-based processing
    # - Cloudinary Video API
    # - AWS Elemental MediaConvert
    # - Google Cloud Video Intelligence API
    
    job_data = {
        "source_url": video_url,
        "segments": segments,
        "output_format": "vertical" if make_vertical else "original",
        "quality": "1080p",
        "codec": "h264"
    }
    
    return job_data

# ----------
# Browser-based FFmpeg Processing
# ----------

def generate_ffmpeg_wasm_script(video_url: str, segments: list, make_vertical: bool) -> str:
    """Generate JavaScript code for browser-based video processing using FFmpeg.wasm."""
    
    segments_js = json.dumps([{
        "start": srt_time_to_seconds(seg["start"]),
        "end": srt_time_to_seconds(seg["end"]),
        "text": seg["text"]
    } for seg in segments])
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Video Processing</title>
    <script src="https://unpkg.com/@ffmpeg/ffmpeg@0.12.7/dist/umd/ffmpeg.js"></script>
    <script src="https://unpkg.com/@ffmpeg/util@0.12.1/dist/index.js"></script>
</head>
<body>
    <div id="status">Initializing FFmpeg...</div>
    <video id="output" controls style="display:none; max-width:100%;"></video>
    <a id="download" style="display:none;">Download Video</a>
    
    <script>
        const {{ FFmpeg }} = FFmpegWASM;
        const {{ fetchFile, toBlobURL }} = FFmpegUtil;
        
        const segments = {segments_js};
        const videoUrl = "{video_url}";
        const makeVertical = {str(make_vertical).lower()};
        
        async function processVideo() {{
            const ffmpeg = new FFmpeg();
            const statusDiv = document.getElementById('status');
            
            ffmpeg.on('log', ({{ message }}) => {{
                console.log(message);
                statusDiv.textContent = message;
            }});
            
            ffmpeg.on('progress', ({{ progress, time }}) => {{
                statusDiv.textContent = `Processing: ${{Math.round(progress * 100)}}%`;
            }});
            
            try {{
                statusDiv.textContent = 'Loading FFmpeg...';
                
                const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd';
                const coreURL = await toBlobURL(`${{baseURL}}/ffmpeg-core.js`, 'text/javascript');
                const wasmURL = await toBlobURL(`${{baseURL}}/ffmpeg-core.wasm`, 'application/wasm');
                
                await ffmpeg.load({{
                    coreURL,
                    wasmURL,
                }});
                
                statusDiv.textContent = 'Downloading video...';
                
                // Fetch the source video
                const videoData = await fetchFile(videoUrl);
                await ffmpeg.writeFile('input.mp4', videoData);
                
                statusDiv.textContent = 'Creating clips...';
                
                // Create individual clips
                const clipFiles = [];
                for (let i = 0; i < segments.length; i++) {{
                    const segment = segments[i];
                    const clipName = `clip_${{i}}.mp4`;
                    
                    await ffmpeg.exec([
                        '-i', 'input.mp4',
                        '-ss', segment.start.toString(),
                        '-t', (segment.end - segment.start).toString(),
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-preset', 'ultrafast',
                        clipName
                    ]);
                    
                    clipFiles.push(clipName);
                }}
                
                statusDiv.textContent = 'Concatenating clips...';
                
                // Create concat file
                const concatContent = clipFiles.map(file => `file '${{file}}'`).join('\\n');
                await ffmpeg.writeFile('concat.txt', concatContent);
                
                // Concatenate clips
                await ffmpeg.exec([
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', 'concat.txt',
                    '-c', 'copy',
                    'concatenated.mp4'
                ]);
                
                let finalOutput = 'concatenated.mp4';
                
                if (makeVertical) {{
                    statusDiv.textContent = 'Converting to vertical format...';
                    
                    await ffmpeg.exec([
                        '-i', 'concatenated.mp4',
                        '-vf', 'crop=ih*9/16:ih,scale=1080:1920',
                        '-c:a', 'copy',
                        'vertical.mp4'
                    ]);
                    
                    finalOutput = 'vertical.mp4';
                }}
                
                statusDiv.textContent = 'Finalizing...';
                
                // Read the final output
                const data = await ffmpeg.readFile(finalOutput);
                const blob = new Blob([data.buffer], {{ type: 'video/mp4' }});
                const url = URL.createObjectURL(blob);
                
                // Show video and download link
                const video = document.getElementById('output');
                const download = document.getElementById('download');
                
                video.src = url;
                video.style.display = 'block';
                
                download.href = url;
                download.download = 'stitched_clip.mp4';
                download.textContent = 'Download Your Clip';
                download.style.display = 'block';
                
                statusDiv.textContent = 'Complete! Your video is ready.';
                
            }} catch (error) {{
                console.error('Error:', error);
                statusDiv.textContent = `Error: ${{error.message}}`;
            }}
        }}
        
        // Start processing when page loads
        processVideo();
    </script>
</body>
</html>
    """

# ----------
# Main Helper Functions
# ----------

def get_api_key() -> str:
    """Retrieve the OpenAI API key from Streamlit secrets or environment."""
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    return os.getenv("OPENAI_API_KEY", "")

def get_system_prompt() -> str:
    """Returns the system prompt for the AI."""
    return '''
You are an expert viral video editor. Analyze the provided SRT transcript and select compelling segments for a viral short video.

REQUIREMENTS:
1. Select 2-5 separate segments that tell a cohesive story
2. Total duration: 20-60 seconds
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
'''

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
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"AI analysis error: {e}")
        return {}

# ----------
# Streamlit App
# ----------

def main():
    st.set_page_config(page_title="Cloud Video Processor", layout="wide")
    
    st.title("🎬 Cloud Video Processor")
    st.markdown("Process videos directly from Google Drive - **No downloads required!**")
    
    # Method selection
    processing_method = st.radio(
        "Choose Processing Method:",
        ["Browser-based (FFmpeg.wasm)", "Cloud API Integration", "External Service"],
        help="Different methods for handling large files without local download"
    )
    
    # Initialize session state
    if 'processor_state' not in st.session_state:
        st.session_state.processor_state = {
            "video_url": None,
            "video_metadata": None,
            "srt_content": None,
            "analysis_result": None,
            "processing_html": None
        }
    
    state = st.session_state.processor_state
    
    # API Key check
    api_key = get_api_key()
    if not api_key:
        st.error("❌ OpenAI API key required. Add to Streamlit secrets.")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    
    # Sidebar inputs
    st.sidebar.header("📥 Inputs")
    
    drive_url = st.sidebar.text_input(
        "🔗 Google Drive Video URL",
        placeholder="https://drive.google.com/file/d/...",
        help="Paste the shareable Google Drive link"
    )
    
    srt_file = st.sidebar.file_uploader("📄 SRT Transcript", type=["srt"])
    
    make_vertical = st.sidebar.checkbox("📱 Vertical Format (9:16)", value=True)
    
    # Process Google Drive URL
    if drive_url and not state["video_url"]:
        try:
            file_id = extract_drive_file_id(drive_url)
            video_url = get_direct_video_url(file_id)
            
            with st.spinner("Getting video info..."):
                metadata = get_video_metadata_from_url(video_url)
                
            if metadata and metadata["streamable"]:
                state["video_url"] = video_url
                state["video_metadata"] = metadata
                st.success(f"✅ Video ready! Size: {metadata['size_mb']:.1f}MB")
            else:
                st.error("❌ Video not accessible. Make sure it's shared publicly.")
                
        except Exception as e:
            st.error(f"❌ Error processing Drive URL: {e}")
    
    # Load SRT file
    if srt_file and not state["srt_content"]:
        try:
            state["srt_content"] = srt_file.getvalue().decode("utf-8")
            st.success("✅ SRT transcript loaded")
        except Exception as e:
            st.error(f"❌ Error reading SRT: {e}")
    
    # Main interface
    if not (state["video_url"] and state["srt_content"]):
        st.info("👆 Please provide both a Google Drive video URL and SRT file to continue.")
        st.stop()
    
    # Show video info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎥 Source Video")
        # Use the direct URL for video display
        st.video(state["video_url"])
        
    with col2:
        st.subheader("📊 Video Info")
        if state["video_metadata"]:
            st.metric("File Size", f"{state['video_metadata']['size_mb']:.1f} MB")
            st.metric("Type", state["video_metadata"]["content_type"])
            st.success("✅ Streamable")
    
    # Show transcript
    with st.expander("📄 View Transcript"):
        st.text_area("SRT Content", state["srt_content"], height=200)
    
    # AI Analysis
    if not state["analysis_result"]:
        if st.button("🤖 Analyze Transcript", type="primary"):
            with st.spinner("AI is analyzing the transcript..."):
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
        st.subheader("🎯 AI Analysis Results")
        
        analysis = state["analysis_result"]
        
        st.success(f"**Title:** {analysis.get('title', 'Untitled')}")
        st.info(f"**Strategy:** {analysis.get('reason', 'No reason provided')}")
        
        st.subheader("📝 Selected Segments")
        segments = analysis.get("segments", [])
        
        total_duration = 0
        for i, seg in enumerate(segments, 1):
            start_sec = srt_time_to_seconds(seg["start"])
            end_sec = srt_time_to_seconds(seg["end"])
            duration = end_sec - start_sec
            total_duration += duration
            
            st.markdown(f"**Segment {i}:** `{seg['start']} → {seg['end']}` ({duration:.1f}s)")
            st.write(f"*\"{seg['text']}\"*")
        
        st.metric("Total Duration", f"{total_duration:.1f} seconds")
        
        # Processing options
        st.subheader("🚀 Process Video")
        
        if processing_method == "Browser-based (FFmpeg.wasm)":
            if st.button("🎬 Generate Video (Browser)", type="primary"):
                with st.spinner("Generating processing script..."):
                    html_script = generate_ffmpeg_wasm_script(
                        state["video_url"], 
                        segments, 
                        make_vertical
                    )
                    state["processing_html"] = html_script
                    st.rerun()
                    
        elif processing_method == "Cloud API Integration":
            st.info("🔧 **Coming Soon:** Direct integration with cloud video processing APIs")
            
            if st.button("📋 Show API Payload", type="secondary"):
                job_data = create_video_processing_job(state["video_url"], segments, make_vertical)
                st.json(job_data)
                
        else:  # External Service
            st.info("🌐 **Manual Processing:** Use external tools with these parameters")
            
            with st.expander("📋 Processing Instructions"):
                st.markdown("### For Manual Processing:")
                st.markdown(f"1. **Source:** `{state['video_url']}`")
                st.markdown("2. **Segments to extract:**")
                
                for i, seg in enumerate(segments, 1):
                    st.markdown(f"   - Clip {i}: {seg['start']} to {seg['end']}")
                
                st.markdown("3. **Post-processing:**")
                st.markdown("   - Concatenate clips in order")
                if make_vertical:
                    st.markdown("   - Convert to 9:16 aspect ratio (1080x1920)")
                st.markdown("   - Export as MP4 with H.264 codec")
    
    # Show browser processor
    if state.get("processing_html"):
        st.subheader("🌐 Browser-based Video Processor")
        st.markdown("**The video will be processed entirely in your browser - no server uploads!**")
        
        # Embed the HTML processor
        st.components.v1.html(state["processing_html"], height=600, scrolling=True)
        
        st.warning("⚠️ **Note:** Large videos may take several minutes to process. Keep this tab open until complete.")
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Start Over"):
        st.session_state.processor_state = {
            "video_url": None,
            "video_metadata": None,
            "srt_content": None,
            "analysis_result": None,
            "processing_html": None
        }
        st.rerun()

if __name__ == "__main__":
    main()
