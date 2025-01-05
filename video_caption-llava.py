import av
import torch
import numpy as np
import os
import csv
import re
from pathlib import Path
# from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def initialize_model():
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    return model, processor


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)  # Seek to the beginning of the video
    start_index = indices[0]  # Get the starting frame index
    end_index = indices[-1]  # Get the ending frame index
    for i, frame in enumerate(container.decode(video=0)):  # Decode video frames, i: the index of the current frame, frame: the current frame being decoded
        if i > end_index:  # Stop if the current frame index exceeds the end index
            break
        if i >= start_index and i in indices:  # Check if the current frame index is within the range and in the indices list
            frames.append(frame)  # Append the frame to the frames list
    rgb_frames = np.stack([frame.to_ndarray(format="rgb24") for frame in frames])  # Convert frames to RGB format and stack them into a numpy array
    
    return rgb_frames


def process_video(video_path, model, processor, csv_path):
    '''
    Process a video file and save the caption to a CSV file.
    Sampling frames from the video involves taking the total length of the video, dividing it into 8 equal segments, 
    sampling one frame from each segment, and processing all 8 frames together in a single batch.
    Args:
        video_path (Path): The path to the video file.
        model (LlavaNextVideoForConditionalGeneration): The model to use for video captioning.
        processor (LlavaNextVideoProcessor): The processor to use for video captioning.
        csv_path (Path): The path to the CSV file to save the captions.
    '''
    try:
        container = av.open(str(video_path)) # Open the video file
        
        # Create conversation prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe what is happening in this video. Be consice, yet detailed."},
                    {"type": "video"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Sample frames
        total_frames = container.streams.video[0].frames # Get the total number of frames in the video
        indices = np.arange(0, total_frames, total_frames / 8).astype(int) # Sample 8 frames from the video, we can sample more frames for longer videos
        clip = read_video_pyav(container, indices) # Read the sampled frames
        
        # Process video and generate description
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
        output = model.generate(**inputs_video, max_new_tokens=200, do_sample=False, num_beams=3) # deterministic generation
        full_response = processor.decode(output[0], skip_special_tokens=True)
        
        assistant_response = full_response.split("ASSISTANT: ")[-1]  # Get everything after "ASSISTANT: "
        # use regex to remove the "In the video, we see" prefix
        description = re.sub(r'^In the video,\s+we see\s+', '', assistant_response)
        
        # Save description to text file
        output_path = video_path.with_suffix('.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(description)
        
        # Save to CSV
        save_to_csv(csv_path, video_path, description)
            
        logger.info(f"Processed: {video_path.name}")
        container.close()
        
    except Exception as e:
        logger.error(f"Error processing {video_path.name}: {str(e)}")


def save_to_csv(filename, video_name, caption):
    """Save caption to CSV file"""
    
    # Check if file exists
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header for new file
        if not file_exists:
            writer.writerow(['video_name', 'caption'])
        
        writer.writerow([os.path.basename(video_name), caption]) # Write the video name(not full path) and caption to the CSV file


def main():
    # Get video directory from user
    while True:
        video_dir = input("Enter the directory path containing the videos: ").strip()

        # Remove quotes if present
        video_dir = video_dir.replace('"', '').replace("'", '').replace('`', '')

        if os.path.isdir(video_dir):
            break
        logger.error("Invalid directory path. Please try again.")
    
    # Initialize model and processor
    logger.info("Initializing model...")
    model, processor = initialize_model()
    
    # Process all videos in directory
    video_dir = Path(video_dir)
    video_files = list(video_dir.glob('*.mp4'))
    
    if not video_files:
        logger.warning("No MP4 files found in the specified directory.")
        return
    
    # Create CSV file path in the same directory as the videos
    csv_path = video_dir / 'video_captions.csv'
    logger.info(f"Saving captions to: {csv_path}")
    
    logger.info(f"Found {len(video_files)} MP4 files. Starting processing...")
    
    for video_path in video_files:
        process_video(video_path, model, processor, csv_path)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
