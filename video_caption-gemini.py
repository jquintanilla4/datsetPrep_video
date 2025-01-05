import os
import time
import random
from dotenv import load_dotenv
import google.generativeai as genai
import os.path
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define the prompt to be used for generating descriptions and prompts
prompt = """Please follow this two-step process:

First, internally analyze the video frames by creating a detailed description including:
- All characters and their expressions
- Actions and positioning
- Environmental elements
- Colors and lighting
- Objects and their relationships
- Overall scene composition

Second, based on your internal description, provide ONLY a comprehensive caption that includes:
- Character features and expressions (facial features, emotions, distinctive traits)
- Positioning and actions (pose, gestures, movement)
- Environmental details (surroundings, objects, terrain)
- Color information (specific colors of elements)
- Spatial relationships (composition, placement of elements)
- Scene composition (framing, depth, perspective)
- Time of day lighting conditions (natural or artificial light sources)
- Shadow details (cast shadows, ambient occlusion)
- Weather implications (environmental conditions)

Important: Do not include your internal description in the output - only provide the final descriptive caption.

The final caption should only be one paragraph."""

# Define the system instruction (including the one-shot example)
system_instruction = """
You're a film director and cinematographer.
An expert in describing videos and all of their details.
Your task is to first internally analyze each video in detail,
then provide only a comprehensive prompt based on that analysis.
Never reference other videos in your prompts.
Each prompt must be completely self-contained.
"""

# Configure the generation settings for the Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the Gemini model with the specified configuration and system instructions
model = genai.GenerativeModel(
    # model_name="gemini-1.5-flash",
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=system_instruction,
)


def process_single_video(video_path, model, prompt):
    """Process a single video and generate analysis."""
    try:
        logger.info(f"Uploading video: {video_path}")
        video_file = genai.upload_file(path=video_path)
        logger.info(f"Completed upload: {video_file.uri}")

        # Wait for video processing to complete
        while video_file.state.name == "PROCESSING":
            logger.info('.', end='')
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")

        messages = [video_file, prompt]
        response = generate_with_retry(model, messages)

        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.txt"
        output_path = os.path.join(os.path.dirname(video_path), output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text.strip())

        return True

    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        return False

def process_folder(folder_path, model, prompt):
    """Process all videos in a folder one at a time."""
    video_files = [
        filename for filename in os.listdir(folder_path)
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))  # Updated for video extensions
    ]

    with tqdm(total=len(video_files), desc="Processing videos") as pbar: # Progress bar for video folder processing
        for filename in video_files:
            video_path = os.path.join(folder_path, filename)
            
            base_name = os.path.splitext(filename)[0]
            processed_path = os.path.join(folder_path, f"{base_name}.txt")
            
            if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
                logger.info(f"Skipping {filename}: Processed file already exists")
                pbar.update(1)
                continue

            process_single_video(video_path, model, prompt)
            logger.info(f"Processed {filename}")
            
            pbar.update(1)


def generate_with_retry(model, messages, max_retries=5):
    """
    Generates content using the provided generative model, with retry logic for handling exceptions.

    Args:
        model: The generative model to use for generating content.
        messages: The input messages for content generation, including image data and text prompt.
        max_retries: The maximum number of retries in case of exceptions.

    Returns:
        The raw response object, or raises an exception after retries.
    """
    retry_count = 0
    base_delay = 10
    delay_multiplier = 2.5  # Multiplier for exponential backoff
    request_options = {"timeout": 120.0}  # Timeout for each API request

    while retry_count < max_retries:  # Loop until max retries reached
        try:
            start_time = time.time()  # Record start time for API call
            
            response = model.generate_content( # Generate content using the provided model and messages
                messages,
                request_options=request_options  # Options for the request
            )
            logger.info(f" API call took {time.time() - start_time:.2f} seconds")

            # Return the raw response object
            return response  # Successful response returned

        except (ResourceExhausted, DeadlineExceeded) as e:  # Handle specific exceptions
            logger.error(f"Attempt {retry_count + 1} failed. Error: {e}")
            retry_count += 1  # Increment retry counter

            if retry_count == max_retries:  # Check if max retries reached
                raise  # Raise exception if retries exhausted

            # Calculate delay with exponential backoff and random jitter
            delay = base_delay * (delay_multiplier ** retry_count) + \
                random.uniform(0, base_delay * retry_count)  # Calculate delay
            logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)  # Wait for the calculated delay before retrying


def main():
    # Get folder path from user
    folder_path = input("Enter the directory path containing the videos: ").strip()

    # Clean the input path
    folder_path = folder_path.replace('"', '').replace("'", '').replace('`', '')

    if not os.path.isdir(folder_path):
        logger.error(f"Error: '{folder_path}' is not a valid directory.")
        return

    # Process the folder
    process_folder(folder_path, model, prompt)


if __name__ == "__main__":
    main()
