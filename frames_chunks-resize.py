import cv2
import os
import inquirer
from tqdm import tqdm
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def resize_frame(frame, target_size):
    height, width = frame.shape[:2]

    # Always resize height to target size and adjust width proportionally
    new_size = (int(width * (target_size / height)), target_size)
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)


def process_video_opencv(video_path, target_size, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate new dimensions for the first frame to set up the video writer
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Error: Could not read the first frame")
        cap.release()
        return

    resized_frame = resize_frame(first_frame, target_size)
    new_height, new_width = resized_frame.shape[:2]

    # Create video writer
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, fps, (new_width, new_height))

    # Reset video capture to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process all frames
    with tqdm(total=total_frames, desc='Processing video') as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = resize_frame(frame, target_size)
            # write the resized frame to the new resized video in the same directory
            out.write(resized_frame)
            bar.update(1)

    cap.release()
    out.release()
    logger.info("Video processing completed!")


def process_video_ffmpeg(video_path, target_size, output_path):
    try:
        # Specify the full path to system FFmpeg
        FFMPEG_PATH = '/usr/bin/ffmpeg'  # run which ffmpeg to find the path
        subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH")
        raise RuntimeError("FFmpeg is required but not found")

    # Calculate the new width while maintaining aspect ratio
    probe_cmd = [
        FFMPEG_PATH,
        '-i', video_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams'
    ]

    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg probe error: {result.stderr}")
        return

    # FFmpeg command for resizing with audio
    cmd = [
        FFMPEG_PATH,
        '-y',  # Overwrite output file if it exists
        '-i', video_path,
        '-vf', f'scale=-2:{target_size}',  # -2 maintains aspect ratio
        '-c:v', 'libx264',  # Use H.264 codec
        '-crf', '19',  # Quality setting (lower = better quality)
        '-c:a', 'copy',  # Copy audio stream without re-encoding
        output_path
    ]

    logger.info("Processing video with FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        return

    logger.info("Video processing completed!")


# process the video router function
def process_video(video_path, target_size, output_dir, keep_audio=False):
    filename, ext = os.path.splitext(os.path.basename(video_path))
    output_path = os.path.join(output_dir, f"{filename}_{target_size}px{ext}")

    if keep_audio:
        process_video_ffmpeg(video_path, target_size, output_path)
    else:
        process_video_opencv(video_path, target_size, output_path)


def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def main():
    questions = [
        inquirer.Path('video_path',
                      message="Enter the directory path containing the videos",
                      path_type=inquirer.Path.DIRECTORY,
                      exists=True),
        inquirer.List('height',
                      message="Select target height",
                      choices=['1024', '960', '768', '512', '480', '360', '256', '128'],
                      carousel=True),
        inquirer.List('audio',
                      message="Select audio preference",
                      choices=['keep audio', 'no audio'],
                      carousel=True)
    ]

    # it prompts the user for the video path and target height
    answers = inquirer.prompt(questions)
    video_path = answers['video_path']
    target_size = int(answers['height'])

    # converts the audio choice to a boolean
    keep_audio = answers['audio'] == 'keep audio'

    # Remove whitespace and quotes if present
    video_path = video_path.strip().replace(
        '"', '').replace("'", '').replace('`', '')

    # Create resized subfolder
    output_dir = os.path.join(video_path, "chunks_resized")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all video files in the directory
    for filename in os.listdir(video_path):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(video_path, filename)
            logger.info(f"\nProcessing {filename}...")

            # Get original dimensions
            _, original_height = get_video_dimensions(full_path)
            if original_height is None:
                continue

            # Validate target size
            if target_size > original_height:
                logger.error(f"Target height ({target_size}px) is larger than original height ({original_height}px). Skipping {filename}")
                continue
            elif target_size == original_height:
                logger.info(f"Video is already at target height ({original_height}px). Skipping {filename}")
                continue

            process_video(full_path, target_size, output_dir, keep_audio)


if __name__ == "__main__":
    main()
