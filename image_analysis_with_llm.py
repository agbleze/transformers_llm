
import base64
from pathlib import Path

def to_data_url(path, mime="image/jpeg"):
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"