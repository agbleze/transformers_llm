
#%%
import eval_type_backport
from eval_type_backport import install_patch

install_patch()

import base64
from IPython.display import HTML
from IPython.display import Video
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import librosa
import audioread
from IPython.display import Video
from IPython.display import Audio
import os, subprocess, sys, numpy as np
import soundfile as sf
from IPython.display import Audio, display
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch
import urllib.request
import time

#%%
def inference_audio(audio_waveform, sampling_rate,
                    prompt, sys_prompt="You are a helpful assistant.",
                    max_frames=32,
                    model_id="Qwen/Qwen2.5-Omni-7B",
                    ):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id,
                                                                torch_type="auto",
                                                                device_map="auto",
                                            )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    messages = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}, 
                {"role": "user", "content": [{"type": "audio", "audio": audio_waveform, "sampling_rate": sampling_rate},
                                             {"type": "text", "text": prompt}
                                            ]
                 }
                ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", 
                       padding=True, use_audio_in_video=False
                       ).to(model.device)
    
    for k, v in inputs.items():
        if hasattr(v, "dtype") and v.dtype.is_floating_point:
            inputs[k] = v.to(model.dtype)
            
    with torch.inference_mode():
        output = model.generate(**inputs,
                                use_audio_in_video=False,
                                return_audio=False,
                                max_new_tokens=256
                                )
    out_text = processor.batch_decode(output, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False
                                      )
    return out_text
    