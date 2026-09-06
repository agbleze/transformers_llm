
#%%
#import eval_type_backport
#from eval_type_backport import install_patch

#install_patch()

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
import matplotlib.pyplot as plt

#%%

def download_with_retry(url, out, tries=3, delay=2):
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, out)
            if os.path.getsize(out) > 0:
                return
        except Exception as e:
            if i == tries -1:
                raise
            time.sleep(delay)
#%%
def inference_audio(audio_waveform, sampling_rate,
                    prompt, sys_prompt="You are a helpful assistant.",
                    max_frames=32,
                    model_id="Qwen/Qwen2.5-Omni-7B",
                    ):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id,
                                                                torch_dtype="auto",
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
    
    
if __name__ == "__main__":
    #%%
    video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/music.mp4"
    mp4_path = "audio_source.mp4"
    wav_path = "audio_16k.wav"
    
    #download_with_retry(video_url, mp4_path)
    
    #%%
    # cmd = ["ffmpeg", "-y", "-i", mp4_path, "-vn", "-ac", "1", "-ar", "16000",
    #        "-f", "wav", wav_path
    #        ]
    # subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    audio_16k, sr = sf.read(wav_path, dtype="float32")
    
    #%%
    S = librosa.feature.melspectrogram(y=audio_16k, sr=sr, n_fft=1024,
                                       hop_length=256, n_mels=80
                                       )
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(S_db, x_axis="time", y_axis="mel",
                             sr=sr, hop_length=256
                             )
    plt.title("Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    #%%
    sys_prompt = "You analyze only the audio. Ignore visuals. Be concise."
    prompt = "Identify the main instruments, tempo feel, time signature if clear, and likely genre in bullet points. Then explain why your answers."

    response = inference_audio(audio_16k, sr, prompt, sys_prompt=sys_prompt)
    print("Model Response:\n", response[0])
    
# %%
