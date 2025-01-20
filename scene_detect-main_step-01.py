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
FFMPEG_CONFIGS = {
    'nvidia_4090': {
        'DOWNSCALE_FACTOR': 1,
        'THREADS': 32,
        'ADAPTIVE_THRESHOLD': 2.0,
        'MIN_SCENE_LEN': None,  # User will specify the minimum scene length
        'MIN_CONTENT_VAL': 15.0,
        'FRAME_WINDOW': 2,
        'VIDEO_TEMPLATE': 'cut_{:03d}.mp4',
        'FFMPEG_INPUT_PARAMS': [  # Parameters that must come before input
            '-hwaccel', 'cuda',
            '-hwaccel_device', '0',
            '-extra_hw_frames', '8'
        ],
        'FFMPEG_OUTPUT_PARAMS': [  # Parameters for output/encoding
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',
            '-tune', 'hq',
            '-rc', 'vbr_hq',
            '-cq', '20',
            '-b:v', '20M',
            '-maxrate', '30M',
            '-bufsize', '30M',
            '-profile:v', 'high',
            '-spatial-aq', '1',
            '-temporal-aq', '1',
            '-aq-strength', '8',
            '-surfaces', '32',
            '-multipass', 'fullres',
            '-gpu', '0'
        ]
    },
    'cpu': {
        'DOWNSCALE_FACTOR': 2,
        'THREADS': 8,
        'ADAPTIVE_THRESHOLD': 2.0,
        'MIN_SCENE_LEN': None,
        'MIN_CONTENT_VAL': 15.0,
        'FRAME_WINDOW': 2,
        'VIDEO_TEMPLATE': 'cut_{:03d}.mp4',
        'FFMPEG_INPUT_PARAMS': [],  # No special input parameters for CPU
        'FFMPEG_OUTPUT_PARAMS': [
            '-c:v', 'libx264',
            '-crf', '19',
            '-c:a', 'copy'
        ]
    }
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


def detect_scene_chunks(video_path, config):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Add file size check
    file_size = os.path.getsize(video_path)
    if file_size > 4 * 1024 * 1024 * 1024:  # 4GB
        logger.warning("Large video file detected. This might require significant memory.")

    logger.info(f"Starting cut detection on: {video_path}")

    try:
        # Create video stream
        video = open_video(video_path, backend='opencv')
        framerate = video.frame_rate
        logger.info(f"Framerate: {framerate}")

        # Detect scenes with built-in progress bar, main detection code block
        scenes = detect(video_path, AdaptiveDetector(
            adaptive_threshold=config['ADAPTIVE_THRESHOLD'],
            min_scene_len=config['MIN_SCENE_LEN'],
            min_content_val=config['MIN_CONTENT_VAL'],
            window_width=config['FRAME_WINDOW'],
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
    """Export a single video chunk using FFmpeg.

    Args:
        args (tuple): A tuple containing:
            - video_path (str): Path to the source video file
            - output_path (str): Path where the exported chunk should be saved
            - start_time (float): Start time in seconds for the chunk
            - duration (floa t): Duration in seconds for the chunk
            - config (dict): FFMPEG configuration dictionary

    Returns:
        tuple: A tuple containing (output_path, success_status)
    """
    video_path, output_path, start_time, duration, config = args

    start_timecode = str(datetime.timedelta(seconds=start_time))
    duration_timecode = str(datetime.timedelta(seconds=duration))

    # Check if video has audio stream with more detailed probe
    probe_cmd = [
        'ffprobe',  # Use ffprobe directly for stream detection
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=nw=1:nk=1',
        video_path
    ]

    try:
        has_audio = 'audio' in subprocess.check_output(probe_cmd, text=True).strip()
        logger.info(f"Audio stream detection: {'Found' if has_audio else 'Not found'}")
    except subprocess.CalledProcessError:
        logger.warning(f"Failed to detect audio streams, assuming no audio")
        has_audio = False

    # Base command
    cmd = [
        FFMPEG_PATH,
        '-y',
        '-loglevel', 'warning'
    ]

    # Add input parameters from config
    cmd.extend(config['FFMPEG_INPUT_PARAMS'])

    # Add input file and timing parameters
    cmd.extend([
        '-i', video_path,
        '-ss', start_timecode,
        '-t', duration_timecode,
        '-map', '0:v:0'  # Always map video stream
    ])

    # Add output parameters from config
    cmd.extend(config['FFMPEG_OUTPUT_PARAMS'])

    # Add audio mapping and parameters if present
    if has_audio:
        cmd.extend([
            '-map', '0:a:0?',  # Map audio stream if present, ? makes it optional
            '-c:a', 'aac',     # Use AAC codec for audio
            '-b:a', '192k'     # Set audio bitrate
        ])
    else:
        cmd.extend(['-an'])    # No audio

    # Add final output path
    cmd.append(output_path)

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

        if result.stderr:
            logger.debug(f"FFmpeg stderr output: {result.stderr}")

        return output_path, True

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for {output_path}:")
        logger.error(f"Error output: {e.stderr}")
        return output_path, False
    except Exception as e:
        logger.error(f"Unexpected error while processing {output_path}: {str(e)}")
        return output_path, False


def retry_failed_exports(video_path, output_dir, failed_exports, scenes, framerate, config, max_retries=3):
    """Retry failed exports with improved error handling and scene matching logic.

    Args:
        video_path (str): Path to the source video
        output_dir (str): Directory for output files
        failed_exports (list): List of failed export paths
        scenes (list): List of detected scenes
        framerate (float): Video framerate
        config (dict): FFMPEG configuration
        max_retries (int, optional): Maximum retry attempts. Defaults to 3.
    """
    logger.info(f"Attempting to retry {len(failed_exports)} failed exports...")

    for retry in range(max_retries):
        if not failed_exports:
            break

        logger.info(f"Retry attempt {retry + 1}/{max_retries}")
        retry_tasks = []

        # Process each failed export
        for failed_path in failed_exports:
            try:
                # Extract the scene number from the failed path
                scene_num = int(os.path.basename(failed_path).split('_')[1].split('.')[0])

                # Validate scene number is within range
                if scene_num < 1 or scene_num > len(scenes):
                    logger.error(f"Invalid scene number {scene_num} for {failed_path}")
                    continue

                # Reconstruct the output path using output_dir and scene number
                output_path = os.path.join(output_dir, config['VIDEO_TEMPLATE'].format(scene_num))

                # Get scene timestamps (0-based index, so subtract 1 from scene_num)
                scene = scenes[scene_num - 1]
                start_time = scene[0].get_frames() / framerate
                duration = (scene[1].get_frames() - scene[0].get_frames()) / framerate

                # Validate timestamps
                if start_time < 0 or duration <= 0:
                    logger.error(f"Invalid timestamps for {failed_path}: start={start_time}, duration={duration}")
                    continue

                retry_tasks.append((
                    video_path,
                    output_path,  # Use the reconstructed output path
                    start_time,
                    duration,
                    config
                ))

            except (ValueError, IndexError) as e:
                logger.error(f"Failed to process {failed_path}: {str(e)}")
                continue

        if not retry_tasks:
            logger.error("No valid tasks to retry")
            return failed_exports

        # Process retry tasks
        still_failed = []
        with tqdm(total=len(retry_tasks), desc=f"Retry attempt {retry + 1}", unit="cut") as pbar:
            with ThreadPoolExecutor(max_workers=config['THREADS']) as executor:
                futures = [executor.submit(export_single_chunk, task) for task in retry_tasks]

                for future, task in zip(futures, retry_tasks):
                    try:
                        output_path, success = future.result()
                        if not success:
                            still_failed.append(output_path)
                            logger.warning(f"Retry failed for {output_path}")
                    except Exception as e:
                        logger.error(f"Retry task failed for {task[1]}: {str(e)}")
                        still_failed.append(task[1])
                    finally:
                        pbar.update(1)

        failed_exports = still_failed
        if failed_exports:
            logger.warning(f"Still failed after attempt {retry + 1}: {len(failed_exports)} cuts")
        else:
            logger.info("All retried exports completed successfully!")
            break

    if failed_exports:
        logger.error("Some exports could not be completed after all retries:")
        for failed in failed_exports:
            logger.error(f"  - {failed}")

    return failed_exports


def export_scene_chunks(video_path, output_dir, config):
    try:
        scenes, framerate = detect_scene_chunks(video_path, config)

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
                output_dir, config['VIDEO_TEMPLATE'].format(i))

            if not os.path.exists(output_path):
                export_tasks.append(
                    (video_path, output_path, start_time, duration, config)
                )

        if not export_tasks:
            logger.info("No new cuts to export")
            return

        # Process all tasks in parallel
        failed_exports = []  # Initialize empty list for failed exports
        max_workers = config['THREADS']
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
                video_path, output_dir, failed_exports, scenes, framerate, config)

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
            inquirer.List('processing_mode',
                          message="Select processing mode",
                          choices=[
                              ('CUDA Nvidia 4090', 'nvidia_4090'),
                              ('CPU Only (Mac/Other)', 'cpu'),
                          ]),
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

        answers = inquirer.prompt(questions)

        if not answers:
            logger.info("Process cancelled by user")
            sys.exit(0)

        # Add answers to variables
        video_path = answers['video_path'].strip().replace('"', '').replace("'", '').replace('`', '')
        CONFIG = FFMPEG_CONFIGS[answers['processing_mode']]
        CONFIG['MIN_SCENE_LEN'] = answers['min_scene_len']

        logger.info(f"Selected video: {video_path}")
        logger.info(f"Processing mode: {answers['processing_mode']}")
        logger.info(f"Using minimum scene length: {CONFIG['MIN_SCENE_LEN']} frames")

        # Setup output directory
        video_dir = os.path.dirname(video_path)
        output_dir = os.path.join(video_dir, 'scene_chunks')
        os.makedirs(output_dir, exist_ok=True)

        # export scenes chunks
        export_scene_chunks(video_path, output_dir, CONFIG)
        logger.info("Processing complete!")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
