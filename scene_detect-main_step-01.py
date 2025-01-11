from scenedetect import detect, open_video
from scenedetect.detectors import AdaptiveDetector
import subprocess
import os
import datetime
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys
import pandas as pd
import inquirer

# Configuration parameters
CONFIG = {
    # Video processing - optimized for speed and quality (reference hardware is 4090)
    'DOWNSCALE_FACTOR': 1,      # 1 for full resolution, 2 for half, 4 for quarter
    'THREADS': 32,              # Increased for better CPU utilization

    # Detection parameters - tuned for better accuracy/speed balance
    'ADAPTIVE_THRESHOLD': 2.0,   # Slightly more aggressive threshold
    'MIN_SCENE_LEN': None,        # Reduced minimum scene length
    'MIN_CONTENT_VAL': 15.0,    # Increased minimum content value
    'FRAME_WINDOW': 2,          # Reduced window size for faster processing

    'VIDEO_TEMPLATE': 'cut_{:03d}.mp4',  # Template for output video filenames
}

# setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# setup ffmpeg path
try:
    FFMPEG_PATH = '/usr/bin/ffmpeg'  # Default path for Linux systems
    subprocess.run([FFMPEG_PATH, '-version'], capture_output=True,
                   check=True)  # Verify FFmpeg is available
except (subprocess.CalledProcessError, FileNotFoundError):
    try:
        FFMPEG_PATH = subprocess.check_output(['which', 'ffmpeg']).decode().strip()  # Try finding ffmpeg in PATH
    except subprocess.CalledProcessError:
        logger.error("FFmpeg is not installed or not in PATH")
        raise RuntimeError("FFmpeg is required but not found")


def detect_scene_chunks(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Starting cut detection on: {video_path}")

    try:
        # Create video stream
        video = open_video(video_path, backend='opencv')
        framerate = video.frame_rate
        logger.info(f"Framerate: {framerate}")

        # Detect scenes with built-in progress bar, main detection code block
        scenes = detect(video_path, AdaptiveDetector(
            adaptive_threshold=CONFIG['ADAPTIVE_THRESHOLD'],
            min_scene_len=CONFIG['MIN_SCENE_LEN'],
            min_content_val=CONFIG['MIN_CONTENT_VAL'],
            window_width=CONFIG['FRAME_WINDOW'],
            luma_only=False
        ), show_progress=True)

        if not scenes:
            logger.warning("No scenes were detected!")
            return [], framerate

        logger.info(f"Found {len(scenes)} cuts")

        # Save to CSV using pandas
        csv_output_path = os.path.join(os.path.dirname(video_path), 'scene_info.csv')
        logger.info(f"Saving scene information to: {csv_output_path}")

        # Create DataFrame with scene information
        scene_data = [
            {
                'Cut Name': f'cut_{i+1:03d}',
                'Start Frame': scene[0].get_frames(),
                'End Frame': scene[1].get_frames(),
                'Length (frames)': scene[1].get_frames() - scene[0].get_frames()
            }
            for i, scene in enumerate(scenes)
        ]

        pd.DataFrame(scene_data).to_csv(csv_output_path, index=False)

        return scenes, framerate

    except Exception as e:
        logger.error(f"Error during scene detection: {str(e)}")
        raise


def export_single_chunk(args):
    """Helper function for parallel processing"""
    video_path, output_path, start_time, duration = args

    start_timecode = str(datetime.timedelta(seconds=start_time))
    duration_timecode = str(datetime.timedelta(seconds=duration))

    # FFmpeg command with optimized NVENC settings for 4090
    cmd = [
        FFMPEG_PATH,
        '-y',
        '-loglevel', 'warning',
        '-hwaccel', 'cuda',
        '-hwaccel_device', '0',
        '-extra_hw_frames', '8',  # Additional hardware frame buffers
        '-i', video_path,
        '-ss', start_timecode,
        '-t', duration_timecode,
        '-c:v', 'h264_nvenc',
        '-preset', 'p1',          # Fastest preset for 4090
        '-tune', 'hq',
        '-rc', 'vbr_hq',         # High quality VBR mode
        '-cq', '20',             # Balanced quality setting
        '-b:v', '20M',           # Increased bitrate
        '-maxrate', '30M',
        '-bufsize', '30M',
        '-profile:v', 'high',
        '-spatial-aq', '1',
        '-temporal-aq', '1',
        '-aq-strength', '8',     # Increased AQ strength
        '-surfaces', '32',
        '-multipass', 'fullres',  # Full resolution multipass
        '-gpu', '0',
        '-c:a', 'copy',
        '-avoid_negative_ts', '1',
        output_path
    ]

    try:
        # First verify the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Exporting chunk to: {output_path}")
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Verify the output file exists and has size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, True
        else:
            logger.error(f"Output file is empty or missing: {output_path}")
            return output_path, False

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for {output_path}:")
        logger.error(f"Error output: {e.stderr}")
        return output_path, False
    except Exception as e:
        logger.error(
            f"Unexpected error while processing {output_path}: {str(e)}")
        return output_path, False


def retry_failed_exports(video_path, output_dir, failed_exports, scenes, framerate, max_retries=3):
    """New function to handle retrying failed exports"""
    logger.info(f"Attempting to retry {len(failed_exports)} failed exports...")

    for retry in range(max_retries):
        if not failed_exports:
            break

        logger.info(f"Retry attempt {retry + 1}/{max_retries}")
        retry_tasks = [(video_path, failed_path,
                       next(scene[0].get_frames() / framerate
                            for i, scene in enumerate(scenes, 1)
                            if os.path.join(output_dir, CONFIG['VIDEO_TEMPLATE'].format(i)) == failed_path),
                       next(scene[1].get_frames() / framerate - scene[0].get_frames() / framerate
                            for i, scene in enumerate(scenes, 1)
                            if os.path.join(output_dir, CONFIG['VIDEO_TEMPLATE'].format(i)) == failed_path))
                       for failed_path in failed_exports]

        still_failed = []
        with tqdm(total=len(retry_tasks), desc=f"Retry attempt {retry + 1}", unit="cut") as pbar:
            with ThreadPoolExecutor(max_workers=CONFIG['THREADS']) as executor:
                futures = [executor.submit(export_single_chunk, task)
                           for task in retry_tasks]

                for future in futures:
                    try:
                        output_path, success = future.result()
                        if not success:
                            still_failed.append(output_path)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Retry task failed: {str(e)}")
                        still_failed.append(output_path)
                        pbar.update(1)

        failed_exports = still_failed
        if failed_exports:
            logger.warning(
                f"Still failed after attempt {retry + 1}: {len(failed_exports)} cuts")
        else:
            logger.info("All retried exports completed successfully!")
            break

    return failed_exports


def export_scene_chunks(video_path, output_dir):
    try:
        scenes, framerate = detect_scene_chunks(video_path)

        if not scenes:
            logger.warning("No cuts detected!")
            return

        # Find the last successfully exported file
        existing_files = [f for f in os.listdir(output_dir)
                          if f.startswith('cut_') and f.endswith('.mp4')]
        last_number = 0
        if existing_files:
            last_number = max(int(f.split('_')[1].split('.')[0])
                              for f in existing_files)
            logger.info(
                f"Found existing files, resuming from cut_{last_number}")

        # Prepare export tasks
        export_tasks = []
        remaining_scenes = scenes[last_number:]

        # Verify source video is still accessible
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Source video no longer accessible: {video_path}")

        # Verify output directory is writable
        test_file = os.path.join(output_dir, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except IOError:
            raise IOError(f"Output directory is not writable: {output_dir}")

        # Check available disk space
        free_space = os.statvfs(output_dir).f_frsize * os.statvfs(output_dir).f_bavail
        if free_space < 1024 * 1024 * 1024:  # Less than 1GB
            logger.warning("Low disk space warning! Less than 1GB available.")

        for i, scene in enumerate(remaining_scenes, start=last_number+1):
            start_time = scene[0].get_frames() / framerate
            end_time = scene[1].get_frames() / framerate
            duration = end_time - start_time
            output_path = os.path.join(
                output_dir, CONFIG['VIDEO_TEMPLATE'].format(i))

            if not os.path.exists(output_path):
                export_tasks.append(
                    (video_path, output_path, start_time, duration)
                )

        if not export_tasks:
            logger.info("No new cuts to export")
            return

        # Process all tasks in parallel
        failed_exports = []  # Initialize empty list for failed exports
        max_workers = CONFIG['THREADS']
        logger.info(
            f"Processing {len(export_tasks)} cuts using {max_workers} workers")

        with tqdm(total=len(export_tasks), desc="Export progress", unit="cut") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(export_single_chunk, task)
                           for task in export_tasks]

                for future in futures:
                    try:
                        output_path, success = future.result()
                        if not success:  # if export_single_chunk returns (path, False)
                            failed_exports.append(output_path)  # add path to failed_exports list
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Export task failed: {str(e)}")
                        failed_exports.append(output_path)  # Also add to list if exception occurs
                        pbar.update(1)

        # After the first complete run, handle any failures
        if failed_exports:
            failed_exports = retry_failed_exports(
                video_path, output_dir, failed_exports, scenes, framerate)

            if failed_exports:
                logger.warning(
                    f"Failed to export {len(failed_exports)} cuts after all retry attempts")
                for path in failed_exports:
                    logger.warning(f"Failed export: {path}")
            else:
                logger.info("All cuts exported successfully after retries")
        else:
            logger.info("All cuts exported successfully on first attempt")

    except KeyboardInterrupt:
        logger.info("Export process interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Export process failed: {str(e)}")
        raise


def main():
    try:
        # Prepare questions for user
        questions = [
            inquirer.Path('video_path',
                          message="Enter the path to your video file",
                          path_type=inquirer.Path.FILE,
                          exists=True),
            inquirer.List('min_scene_len',
                          message="Select minimum scene length (in frames)",
                          choices=[
                              ('30 frames (more sensitive)', 30),
                              ('40 frames', 40),
                              ('50 frames', 50),
                              ('60 frames', 60),
                              ('70 frames (less sensitive)', 70)
                          ],
                          default=30)
        ]

        answers = inquirer.prompt(questions)  # initialize the questions

        if not answers:  # User pressed Ctrl+C
            logger.info("Process cancelled by user")
            sys.exit(0)

        # Add answers to variables
        video_path = answers['video_path']
        video_path = video_path.strip().replace('"', '').replace("'", '').replace('`', '')
        CONFIG['MIN_SCENE_LEN'] = answers['min_scene_len']

        logger.info(f"Selected video: {video_path}")
        logger.info(f"Using minimum scene length: {CONFIG['MIN_SCENE_LEN']} frames")

        # Setup output directory
        video_dir = os.path.dirname(video_path)
        output_dir = os.path.join(video_dir, 'scene_chunks')
        os.makedirs(output_dir, exist_ok=True)

        # export scenes chunks
        export_scene_chunks(video_path, output_dir)
        logger.info("Processing complete!")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
