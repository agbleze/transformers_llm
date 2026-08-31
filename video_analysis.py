
#%%
import base64
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoModelForVision2Seq, AutoProcessor



# %%
model_id = "Qwen/Qwen2.5-Omni-7B"

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id,
                                               torch_dtype="auto",
                                               device_map="auto",
                                               trust_remote_code=True
                                               )
#&&
# %%
processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)
# %%


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