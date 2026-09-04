


def inference_audio(audio_waveform, sampling_rate,
                    prompt, sys_prompt="You are a helpful assistant.",
                    max_frames=32,
                    model_id="Qwen/Qwen2.5-Omni-7B",
                    ):
    messages = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}, 
                {"role": "user", "content": [{"type": "audio", "audio": audio_waveform, "sampling_rate": sampling_rate},
                                             {"type": "text", "text": prompt}
                                            ]
                 }
                ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False).to(model.device))
    