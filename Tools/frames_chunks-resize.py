import cv2
import os
import inquirer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def resize_frame(frame, target_size):
    height, width = frame.shape[:2]
    
    # Always resize height to target size and adjust width proportionally
    new_size = (int(width * (target_size / height)), target_size)
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LANCZOS4)


def process_video(video_path, target_size, output_dir):
    # Create output path with the new subfolder
    filename, ext = os.path.splitext(os.path.basename(video_path)) # it splits the video path into filename and extension
    output_path = os.path.join(output_dir, f"{filename}_{target_size}px{ext}") # it joins the output directory with the filename and extension

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
            out.write(resized_frame) # write the resized frame to the new resized video in the same directory
            bar.update(1)

    logger.info("\nVideo processing completed!")
    cap.release()
    out.release()


def main():
    questions = [
        inquirer.Path('video_path',
                     message="Enter the directory path containing the videos",
                     path_type=inquirer.Path.DIRECTORY,
                     exists=True),
        inquirer.List('height',
                     message="Select target height",
                     choices=['1024', '768', '512'],
                     carousel=True)
    ]
    
    answers = inquirer.prompt(questions) # it prompts the user for the video path and target height
    video_path = answers['video_path']
    target_size = int(answers['height'])
    
    # Remove whitespace and quotes if present
    video_path = video_path.strip().replace('"', '').replace("'", '').replace('`', '')

    # Create resized subfolder
    output_dir = os.path.join(video_path, "resized")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all video files in the directory
    for filename in os.listdir(video_path):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(video_path, filename)
            logger.info(f"\nProcessing {filename}...")
            process_video(full_path, target_size, output_dir)
    

if __name__ == "__main__":
    main()
