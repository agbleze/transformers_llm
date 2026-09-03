


def inference_audio(audio_waveform, sampling_rate,
                    prompt, sys_prompt="You are a helpful assistant.",
                    max_frames=32,
                    model_id="Qwen/Qwen2.5-Omni-7B",
                    ):
    messages = [{"role": "system", "content": []}, 
                {"role": "user", "content": []}
                ]