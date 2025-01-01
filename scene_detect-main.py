# from scenedetect import detect, AdaptiveDetector
from scenedetect.detectors import AdaptiveDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
import subprocess
import os
import datetime
import logging
from tqdm import tqdm

# Configuration parameters
CONFIG = {
    # Video processing
    'DOWNSCALE_FACTOR': 2,      # Factor to scale down video during detection (1 = original size)

    # Detection parameters - for detect-adaptive
    'ADAPTIVE_THRESHOLD': 1.0,   # Threshold for adaptive detection, lower value = more sensitive cut detection
    'MIN_SCENE_LEN': 72,        # Minimum scene length in frames
    'MIN_CONTENT_VAL': 12.0,    # Minimum content value
    'FRAME_WINDOW': 3,          # Window size for adaptive detection
    
    # Output video settings
    'VIDEO_TEMPLATE': 'cut_{:03d}.mp4',
    'VIDEO_CODEC': 'libx264',    
    'VIDEO_CRF': 19,            
}

# setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def detect_scene_chunks(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logger.info(f"Starting cut detection on: {video_path}")
    
    # Create video manager and get framerate
    video = VideoManager([video_path]) # create video manager
    video.set_downscale_factor(CONFIG['DOWNSCALE_FACTOR']) # set downscale factor
    video.start() # start video manager
    try:
        framerate = video.get_framerate() # get framerate
        logger.info(f"Video framerate: {framerate} fps")
    finally:
        video.release()  # Add this to ensure proper cleanup, releases video manager
    
    # Create scene manager with AdaptiveDetector
    scene_manager = SceneManager(StatsManager())  # Initialize the SceneManager with a StatsManager
    scene_manager.add_detector(  # Add an AdaptiveDetector to the scene manager
        AdaptiveDetector(
            adaptive_threshold=CONFIG['ADAPTIVE_THRESHOLD'], # Set the adaptive threshold for detection
            min_scene_len=CONFIG['MIN_SCENE_LEN'], # Set the minimum scene length
            min_content_val=CONFIG['MIN_CONTENT_VAL'], # Set the minimum content value
            window_width=CONFIG['FRAME_WINDOW'], # Set the window width for detection
            luma_only=False # Use both luma and chroma for detection
        )
    )
    
    # Detect scenes
    logger.info("Detecting cuts...")
    scene_manager.detect_scenes(frame_source=video)  # Perform scene detection on the video
    
    # Get scene list and verify lengths
    scenes = scene_manager.get_scene_list()  # Retrieve the list of detected scenes
    logger.info(f"Found {len(scenes)} cuts")
    # Log detailed cut information
    for i, scene in enumerate(scenes):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        length_frames = end_frame - start_frame
        length_seconds = length_frames / framerate
        logger.info(f"Cut {i+1}:")
        logger.info(f"  Start: Frame {start_frame} ({start_frame/framerate:.2f} sec)")
        logger.info(f"  End: Frame {end_frame} ({end_frame/framerate:.2f} sec)")
        logger.info(f"  Length: {length_frames} frames ({length_seconds:.2f} sec)")
    
    return scenes, framerate


def export_scene_chunks(video_path, output_dir):
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not in PATH")
        raise RuntimeError("FFmpeg is required but not found")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    scenes, framerate = detect_scene_chunks(video_path)
    
    if not scenes:
        logger.warning("No cuts detected!")
        return
    
    # Find the last successfully exported file
    existing_files = [f for f in os.listdir(output_dir) 
                     if f.startswith('cut_') and f.endswith('.mp4')]
    last_number = 0
    if existing_files:
        last_number = max(int(f.split('_')[1].split('.')[0]) for f in existing_files)
        logger.info(f"Found existing files, resuming from cut_{last_number}")
    
    # Create progress bar for remaining scenes
    remaining_scenes = scenes[last_number:]
    pbar = tqdm(
        enumerate(remaining_scenes, start=last_number+1),
        total=len(remaining_scenes),
        desc="Exporting cuts",
        unit="cut"
    )
    
    # Process only remaining scenes
    for i, scene in pbar:
        try:
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            duration = end_time - start_time
            
            start_timecode = str(datetime.timedelta(seconds=start_time))
            duration_timecode = str(datetime.timedelta(seconds=duration))
            
            output_path = os.path.join(output_dir, CONFIG['VIDEO_TEMPLATE'].format(i))
            
            # Skip if file already exists
            if os.path.exists(output_path):
                pbar.set_postfix_str(f"Skipping cut {i}, file exists")
                continue
                
            pbar.set_postfix_str(f"Processing cut {i}/{len(scenes)}")
            
            # ffmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', start_timecode,
                '-t', duration_timecode,
                '-c:v', CONFIG['VIDEO_CODEC'],
                '-crf', str(CONFIG['VIDEO_CRF']),
                '-c:a', 'aac',
                '-avoid_negative_ts', '1',
                output_path
            ]
            
            logger.info("Running FFmpeg...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                continue
                
            if os.path.exists(output_path):
                logger.info(f"Successfully exported cut {i} to {output_path}")
            else:
                logger.error(f"Output file was not created: {output_path}")
                
        except Exception as e:
            logger.error(f"Error processing cut {i}: {str(e)}")

def main():
    try:
        # Get video path from user and validate
        while True:
            video_path = input("Enter the path to the video file: ").strip()
            # Remove quotes if present
            video_path = video_path.replace('"', '').replace("'", '').replace('`', '')
            
            if os.path.exists(video_path) and video_path.lower().endswith(('.mp4', '.avi', '.mkv')):
                break
            else:
                logger.error("Invalid video path or unsupported format. Please try again.")
        
        # Setup output directory
        video_dir = os.path.dirname(video_path)
        output_dir = os.path.join(video_dir, 'scene_chunks')
        os.makedirs(output_dir, exist_ok=True)
        # upadate logger
        logger.info(f"Input video: {video_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # export scenes chunks
        export_scene_chunks(video_path, output_dir)
        logger.info("Processing complete!") # update logger
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()