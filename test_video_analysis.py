import base64
import os
import cv2
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image
from openai import OpenAI
from decouple import config

def load_video_to_base64_list(video_path, max_frames=4):
    """
    Uniformly extracts frames using OpenCV, shrinks them to manage bandwidth,
    and encodes them directly to base64 JPEG strings for API transmission.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file missing at: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video container: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video contains 0 frames.")
        
    # Uniform timeline sampling sequence
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    base64_frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            # Convert OpenCV BGR to RGB matrix arrays
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Keep dimensions optimized for quick network uploads
            pil_img = pil_img.resize((512, 512))
            
            # Save matrix memory buffer directly to JPEG byte blocks
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=80)
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_frames.append(f"data:image/jpeg;base64,{b64_str}")
            
    cap.release()
    return base64_frames

def analyze_video_via_api(video_path, prompt, max_frames=4):
    api_key = config("OPENROUTER_API_KEY")
    base_url = config("OPENROUTER_BASEURL")
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # 2. Extract and encode frames locally
    print(f"Extracting and encoding {max_frames} frames from video asset...")
    frame_data_urls = load_video_to_base64_list(video_path, max_frames=max_frames)
    
    # 3. Format payload block following OpenRouter multi-image/video content syntax
    user_content = []
    for url in frame_data_urls:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "low"}
        })
        
    # Append user question directly behind text frames sequence strings
    user_content.append({
        "type": "text",
        "text": f"The images attached are chronological frames sampled from a video timeline. {prompt}"
    })
    
    messages = [{"role": "user", "content": user_content}]
    
    # 4. Target the dynamic free routing layer engine
    model = "openrouter/free"
    
    print("Streaming frame token data to remote execution layer...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=1024,
        temperature=0.2 # Lower temperature guarantees high accuracy for document/text layouts
    )
    
    # 5. Stream output chunks directly to terminal layout
    print("\n--- Model Response ---")
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    video_file = "/home/lin/codebase/__cv_with_roboflow_data/only_disease_pred_video.avi"
    test_prompt = "Please look at the frames carefully and list the names of the authors written on the paper layout."
    
    analyze_video_via_api(video_file, prompt=test_prompt, max_frames=5)
