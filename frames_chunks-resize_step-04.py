import os
import cv2
import inquirer
import subprocess
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def process_video_ffmpeg(video_path, target_size, output_path, keep_audio=False):
    """
    Process video using FFmpeg to resize to target height while maintaining aspect ratio.

    Args:
        video_path (str): Path to input video file
        target_size (int): Target height in pixels
        output_path (str): Path to output directory
        keep_audio (bool, optional): Whether to keep audio in output. Defaults to False.
    """
    try:
        # Specify the full path to system FFmpeg
        FFMPEG_PATH = '/usr/bin/ffmpeg'  # run which ffmpeg to find the path
        subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH")
        raise RuntimeError("FFmpeg is required but not found")

    # FFmpeg command for resizing with progress output
    cmd = [
        FFMPEG_PATH,
        '-y',  # Overwrite output file if it exists
        '-i', video_path,
        '-vf', f'scale=-2:{target_size}',  # -2 maintains aspect ratio
        '-c:v', 'libx264',
        '-crf', '19',
        # '-progress', 'pipe:1',  # Output progress to stdout
    ]

    if keep_audio:
        cmd.extend(['-c:a', 'copy'])  # keep audio
    else:
        cmd.extend(['-an'])  # no audio

    input_filename = os.path.basename(video_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_{target_size}px{ext}"  # Keep the original extension
    full_output_path = os.path.join(output_path, output_filename)
    cmd.append(full_output_path)

    # Run FFmpeg without progress tracking
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        logger.error(f"FFmpeg error: {process.stderr}")
        return False
    return True


def get_video_dimensions(video_path):
    """
    Get the dimensions of a video file.

    Args:
        video_path (str): Path to video file

    Returns:
        tuple: (width, height) in pixels, or (None, None) if video cannot be opened
    """
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
                      choices=['1024', '960', '768', '512', '480', '404', '360', '256', '128'],
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
    output_dir = os.path.join(video_path, f"chunks_resized-{target_size}px")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of video files first
    video_files = [f for f in os.listdir(video_path)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    total_videos = len(video_files)

    # Process all video files with overall progress bar
    with tqdm(total=total_videos, desc="Processing videos", unit="video") as pbar:
        for filename in video_files:  # for each video file
            full_path = os.path.join(video_path, filename)  # get the full path
            logger.info(f"\nProcessing {filename}...")  # log the filename

            # Get original dimensions
            _, original_height = get_video_dimensions(full_path)  # get the original height
            if original_height is None:  # if the original height is None
                pbar.update(1)
                continue  # continue to the next video file

            # Validate target size
            if target_size > original_height:
                logger.error(
                    f"Target height ({target_size}px) is larger than original height ({original_height}px). Skipping {filename}")
                pbar.update(1)
                continue
            elif target_size == original_height:
                logger.info(f"Video is already at target height ({original_height}px). Skipping {filename}")
                pbar.update(1)
                continue

            # Process the video
            process_video_ffmpeg(full_path, target_size, output_dir, keep_audio)
            pbar.update(1)


if __name__ == "__main__":
    main()
