
#%%
import base64
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoModelForVision2Seq, AutoProcessor



# # %%
# model_id = "Qwen/Qwen2.5-Omni-7B"

# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id,
#                                                torch_dtype="auto",
#                                                device_map="auto",
#                                                trust_remote_code=True
#                                                )
# #&&
# # %%
# processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)
# # %%


def load_video(video_path, max_frames=64):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video file contains zero frames.")
    
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
    cap.release()
    return frames


def inference(video_path, prompt, sys_prompt="You are a helpful assistant.",
              max_frames=32
              ):
    model_id = "Qwen/Qwen2.5-Omni-7B"

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id,
                                                torch_dtype="auto",
                                                device_map="auto",
                                                trust_remote_code=True
                                                )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)
    frames = load_video(video_path, max_frames=max_frames)
    
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user",
                 "content": [
                     {"type": "video", "video": frames},
                     {"type": "text", "text": prompt}
                    ]
                 }
                ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_propmt=True)
    
    inputs = processor(text=[text],
                       video=[frames],
                       padding=True,
                       return_tensors="pt"
                       )
    
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024,
                                       do_sample=False
                                       )
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]

    output_text = processor.batch_decode(generated_ids_trimmed,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False
                                         )
    return output_text


if __name__ == "__main__":
    video_file = "/home/lin/codebase/transformers_llm/ch06_screen_recording_attention_is_all_you_need.mp4"  # Replace with your video file path
    test_prompt = "Please describe the content of the video."
    results = inference(video_file, prompt=test_prompt)
    print(results)