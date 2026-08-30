
#%%
import base64
from pathlib import Path
from openai import OpenAI
from decouple import config

#%%
model = "openrouter/free"
#%%
open_router_baseurl = config("OPENROUTER_BASEURL")
open_router_apikey = config("OPENROUTER_API_KEY")
image_path = config("IMAGE_PATH")

#%%
def to_data_url(path, mime="image/jpeg"):
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


client = OpenAI(base_url=open_router_baseurl,
                  api_key=open_router_apikey
                  )

#%%
data_url = to_data_url(image_path)

#%%


response = client.chat.completions.create(model=model,
                                          messages=[{"role": "user",
                                                    "content": [{"type": "image_url",
                                                                "image_url": {"url": data_url, 
                                                                              "detail": "high"
                                                                              }
                                                                },
                                                                {"type": "text",
                                                                "text": "What’s in this image?"
                                                                }
                                                            ]
                                                    }
                                                ],
                                          stream=True,
                                          max_tokens=1024,
                                          temperature=1.0,
                                          top_p=1.0,
                                        #   extra_body={"top_k": 50, "repetition_penalty": 1.0,
                                        #               "min_p": 0.0
                                        #               }
                                          )

#%%
for chunk in response:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="")
        
# %%
