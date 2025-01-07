# This is a jupyter interactive python script
# To run it, turn on the jupyter interactive option in vscode/cursor

import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import re

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
processor = LlavaNextVideoProcessor.from_pretrained(model_id)

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
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
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

video_path = "/home/jquintanilla/Developer/datasets/video_datasets/small_test_sample/cut_067.mp4"
container = av.open(video_path)

# sample uniformly 8 frames from the video, can sample more for longer videos
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)
inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

# original way to get the assistant response
output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# print the raw output
print(processor.decode(output[0], skip_special_tokens=True))

# close the container, free up memory (not necessary in jupyter)
container.close()

# cleaner way to get the assistant response
output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
full_response = processor.decode(output[0], skip_special_tokens=True)
assistant_response = full_response.split("ASSISTANT: ")[-1]  # Get everything after "ASSISTANT: "
print(assistant_response)

# remove the "In the video, we see a" prefix
output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
full_response = processor.decode(output[0], skip_special_tokens=True)
assistant_response = full_response.split("ASSISTANT: ")[-1]  # Get everything after "ASSISTANT: "
# use regex to remove the "In the video, we see" prefix
cleaned_response = re.sub(r'^In the video,\s+we see\s+', '', assistant_response)
print(cleaned_response)