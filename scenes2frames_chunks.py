import cv2
# import os
import math
from pathlib import Path
import subprocess
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def create_video_writer(original_video, output_path, frame_width, frame_height):
    """Create a VideoWriter object with the same properties as the input video"""
    # set parameters
    codec = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = int(original_video.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(
        str(output_path), 
        codec, 
        fps, 
        (frame_width, frame_height)
    )
    return video_writer


def process_video(input_path, output_dir, max_frames):
    """Process a single video into chunks based on max_frames"""
    # Open the video file
    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        logger.error(f"Could not open video {input_path}")
        return

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate number of chunks needed
    num_chunks = math.ceil(total_frames / max_frames)
    
    logger.info(f"\nProcessing {input_path.name}:")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Will be split into {num_chunks} chunks")

    # Create progress bar for chunks
    chunk_pbar = tqdm(range(num_chunks), desc="Processing chunks", unit="chunk")
    
    for chunk_idx in chunk_pbar:
        # Create output filename for this chunk
        output_filename = f"{input_path.stem}_chunk{chunk_idx + 1}{input_path.suffix}"
        output_path = output_dir / output_filename
        
        # Create video writer for this chunk
        writer = create_video_writer(video, output_path, frame_width, frame_height)
        
        # Calculate frames for this chunk
        start_frame = chunk_idx * max_frames
        end_frame = min((chunk_idx + 1) * max_frames, total_frames)
        frames_in_chunk = end_frame - start_frame
        
        chunk_pbar.set_postfix({"frames": f"{start_frame + 1}-{end_frame}"})
        
        # Set video position to start frame
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames for this chunk with progress bar
        frame_pbar = tqdm(range(frames_in_chunk), desc=f"Chunk {chunk_idx + 1}", 
                         leave=False, unit="frame")
        
        for _ in frame_pbar:
            ret, frame = video.read()
            if ret:
                writer.write(frame)
            else:
                break
        
        writer.release()
    
    video.release()


def convert_to_h264(output_dir):
    """Convert all mp4v encoded videos in the output directory to h264"""
    try:
        # Test if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH. Skipping conversion.")
        return False

    logger.info("\nConverting videos to H.264...")
    videos = list(Path(output_dir).glob("*.mp4"))
    
    # Create progress bar for conversion
    for video_path in tqdm(videos, desc="Converting to H.264", unit="video"):
        temp_path = video_path.parent / f"temp_{video_path.name}"
        
        # FFmpeg command for h264 conversion
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-c:v', 'libx264',
            '-crf', '21',
            '-c:a', 'copy',
            str(temp_path)
        ]
        
        try:
            # Hide FFmpeg output since we're using progress bar
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Replace original file with converted file
            video_path.unlink()  # Delete original
            temp_path.rename(video_path)  # Rename temp to original name
            
        except subprocess.CalledProcessError as e:
            logger.error(f"\nError converting {video_path.name}: {e.stderr}")
            if temp_path.exists():
                temp_path.unlink()
            continue
    
    logger.info("\nConversion complete!")
    return True


def main():
    # Get input directory from user
    while True:
        input_dir = input("Enter the path to the directory containing the videos: ").strip()
        
        # Remove quotes if present
        input_dir = input_dir.replace('"', '').replace("'", '').replace('`', '')

        input_path = Path(input_dir)
        if input_path.exists() and input_path.is_dir():
            break
        logger.error("Invalid directory path. Please try again.")

    # Get maximum frames per chunk from user
    while True:
        try:
            max_frames = int(input("Enter the maximum number of frames per chunk: "))
            if max_frames > 0:
                break
            logger.warning("Please enter a positive number.")
        except ValueError:
            logger.warning("Please enter a valid number.")

    # Create output directory
    output_dir = input_path / "frame_chunks"
    output_dir.mkdir(exist_ok=True)

    # Process all MP4 files in the input directory
    mp4_files = list(input_path.glob("*.mp4"))
    
    if not mp4_files:
        logger.warning("No MP4 files found in the specified directory.")
        return

    logger.info(f"\nFound {len(mp4_files)} MP4 files to process.\n")

    # Process each video
    for video_path in mp4_files:
        process_video(video_path, output_dir, max_frames)

    logger.info("All videos have been processed successfully!")
    
    # Ask user if they want to convert to H.264
    if input("\nWould you like to convert the videos to H.264 format? (y/n): ").lower() == 'y':
        convert_to_h264(output_dir)


if __name__ == "__main__":
    main()
