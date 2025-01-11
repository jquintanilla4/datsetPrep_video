import inquirer
from pathlib import Path
import logging
from tqdm import tqdm
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Video format constants
INPUT_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
OUTPUT_VIDEO_CODEC = 'h264'
OUTPUT_VIDEO_EXTENSION = '.mp4'


def get_video_fps(video_path):
    """
    Get the FPS of the input video using FFmpeg.

    Args:
        video_path (str or Path): Path to the input video file

    Returns:
        float: The video's FPS, or None if it couldn't be determined
    """
    try:
        cmd = [
            '/usr/bin/ffmpeg',
            '-i', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # FFmpeg outputs to stderr for this command
        output = result.stderr

        # Look for the fps information in the output
        for line in output.split('\n'):
            if 'fps' in line:
                # Extract fps value using string manipulation
                fps_str = line.split(',')[5].strip().split()[0]
                return float(fps_str)
        return None
    except Exception as e:
        logger.error(f"Error getting FPS for {video_path}: {e}")
        return None


def change_video_fps(video_path, target_fps, convert_h264=False):
    """
    Change the FPS of a video file to the target FPS using FFmpeg.

    Args:
        video_path: Path to the input video file
        target_fps: Desired output FPS
        convert_h264: Whether to convert the output to h264 codec
    """
    # Check current FPS
    current_fps = get_video_fps(video_path)
    if current_fps is None:
        logger.error(f"Could not determine FPS for {video_path}")
        return

    # Skip if FPS is already at target
    if abs(current_fps - target_fps) < 0.01:  # Using small threshold for float comparison
        logger.info(f"Skipping {video_path} - already at target FPS ({current_fps})")
        return

    logger.info(f"Converting {video_path} from {current_fps} FPS to {target_fps} FPS")

    try:
        FFMPEG_PATH = '/usr/bin/ffmpeg'  # run which ffmpeg to find the path
        subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH")
        raise RuntimeError("FFmpeg is required but not found")

    input_path = Path(video_path)
    # Create fps_change directory if it doesn't exist
    output_dir = input_path.parent / 'fps_change'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_{target_fps}fps{OUTPUT_VIDEO_EXTENSION}"

    # FFmpeg command for changing FPS and optionally converting to h264
    cmd = [
        FFMPEG_PATH,
        '-y',  # Overwrite output files without asking
        '-i', str(input_path),
        '-fps_mode', 'cfr',  # Constant frame rate
        '-r', str(target_fps)
    ]

    if convert_h264:
        cmd.extend([
            '-c:v', 'libx264',
            '-crf', '19'  # Quality setting (lower = better quality, 19 is very good)
        ])
    else:
        cmd.extend(['-c:v', 'libx264'])  # Use h264 anyway as it's more compatible

    cmd.append(str(output_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stderr:
            logger.debug(f"FFmpeg output for {input_path.name}: {result.stderr}")

        logger.info(f"Successfully converted {video_path} to {target_fps} FPS")

    except subprocess.CalledProcessError as e:
        logger.error(f"\nError processing {input_path.name}: {e.stderr}")
        if output_path.exists():
            output_path.unlink()


def process_directory(directory, target_fps, convert_h264=False):
    """
    Process all video files in the given directory and change their FPS.

    Args:
        directory: Path to the directory containing video files
        target_fps: Desired output FPS
        convert_h264: Whether to convert the output to h264 codec
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.error("Error: Directory does not exist!")
        return

    videos = [f for f in directory_path.glob('*') if f.suffix.lower() in INPUT_VIDEO_EXTENSIONS]

    if not videos:
        logger.warning("No video files found in the directory!")
        return

    logger.info(f"Found {len(videos)} video files to process")
    for video_path in tqdm(videos, desc="Processing videos", unit="video"):
        change_video_fps(str(video_path), target_fps, convert_h264)


def main():
    """
    Main function to handle user input and initiate video processing.
    """
    # Get directory path and preferences
    questions = [
        inquirer.Text(
            'directory',
            message="Enter the directory path containing the videos"
        ),
        inquirer.List(
            'fps',
            message="Select the target FPS",
            choices=['24', '25']
        ),
        inquirer.Confirm(
            'convert_h264',
            message="Do you want to convert videos to h264 codec?",
            default=False
        )
    ]

    answers = inquirer.prompt(questions)

    if answers:
        directory = answers['directory']
        target_fps = int(answers['fps'])
        convert_h264 = answers['convert_h264']

        logger.info(f"\nProcessing videos in: {directory}")
        logger.info(f"Target FPS: {target_fps}")
        logger.info(f"Convert to h264: {convert_h264}\n")

        process_directory(directory, target_fps, convert_h264)
    else:
        logger.error("Operation cancelled by user")


if __name__ == "__main__":
    main()
