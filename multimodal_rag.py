
import os
import io
import re
import csv
import base64
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, Image as PILImage
import fitz
import torch
import easyocr
from torch import Tensor

from transformers import AutoModelForObjectDetection
from torchvision import transforms
from openai import OpenAI
import qdrant_client
from llama_index.core import (ServiceContext, SimpleDirectoryReader,
                              VectorStoreIndex,
                              StorageContext
                              )
from llama_index.core.schema import (ImageDocument,
                                     ImageNode,
                                     Document
                                     )
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.llms.openai import OpenAI as OpenAIIndex

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.indices.multi_modal.retriever import MultiModalVectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

from typing import Tuple, Union, List, Dict, Any
from decouple import config

device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
pdf_url = "https://arxiv.org/pdf/2505.09388.pdf"
pdf_filename = "Qwen3.pdf"

subprocess.run(["wget", "--user-agent", "Mozilla",
                pdf_url, "-O", pdf_filename
                ], check=True
               )

os.path.exists(pdf_filename)


PDF_PATH = "Qwen3.pdf"

uploaded_pdf_path = Path(PDF_PATH)
output_dir = uploaded_pdf_path.stem

output_path = Path(f"{output_dir}")
output_path.mkdir(parents=True, exist_ok=True)


pdf_document = fitz.open(str(uploaded_pdf_path))
total_pages = pdf_document.page_count

pad_width = len(str(total_pages))

for page_number in range(total_pages):
    page = pdf_document[page_number]
    pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    filename = f"page_{str(page_number + 1).zfill(pad_width)}.png"
    image.save(output_path / filename)
pdf_document.close()


image_paths_sorted = sorted(output_path.glob("page_*.png"))


def extract_page_number(path: Union[str, Path]) -> Union[int, float]:
    path = Path(path)
    match = re.search(f"page_(\d+)\.png", path.name)
    return int(match.group(1) if match else float("inf"))


image_paths_sorted_numeric = sorted(image_paths_sorted, key=extract_page_number)


def plot_images(image_paths, title="Sample PDF Pages"):
    plt.figure(figsize=(16, 9))
    for idx, img_path in enumerate(image_paths[6:10]):
        img = Image.open(img_path)
        plt.subplot(2, 2, idx + 1)
        plt.imshow(img)
        plt.title(img_path.name)
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
plot_images(image_paths_sorted_numeric, title="Pages 6 - 10 of Qwen3.pdf")



# Plot 2 sample pages side by side at larger scale
def plot_two_pages(image_paths, title="Sample PDF Pages"):
    plt.figure(figsize=(16, 10))  # wider and taller
    for idx, img_path in enumerate(image_paths[:2]):  # just first 2 pages for example
        img = Image.open(img_path)
        plt.subplot(1, 2, idx + 1)  # 1 row, 2 columns
        plt.imshow(img)
        plt.title(img_path.name, fontsize=12)
        plt.axis("off")
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()

# Example: show pages 6 and 7 side by side
plot_two_pages(image_paths_sorted_numeric[6:8], title="Pages 6 and 7 of Qwen3.pdf")



# loads the image files into ImageDocument objects for multimodal indexing

document_images = SimpleDirectoryReader("./Qwen3/").load_data()

client = qdrant_client.QdrantClient(path="qdrant_index")

text_store = QdrantVectorStore(client=client,
                                collection_name="text_collection"
                                )

image_store = QdrantVectorStore(client=client, collection_name="image_collection")
storage_context = StorageContext.from_defaults(vector_store=text_store,
                                               image_store=image_store
                                               )

index = MultiModalVectorStoreIndex.from_documents(document_images,
                                                  storage_context=storage_context
                                                  )

retriever_engine = index.as_retriever(image_similarity_top_k=2)


query = "Compare Qwen2.5 and Qwen3."
assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)
retrieval_results = retriever_engine.text_to_image_retrieve(query)


def plot_images(image_paths, title="Retrieved Images"):
    plt.figure(figsize=(15, 6))
    for idx, img_path in enumerate(image_paths[:10]):
        img = Image.open(img_path)
        plt.subplot(2, 5, idx + 1)
        plt.imshow(img)
        plt.title(Path(img_path).name)  # convert str to Path
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
retrieved_images = []
for res_node in retrieval_results:
    if isinstance(res_node, ImageNode):
        retrieved_images.append(res_node.node.metadata["file_path"])
    else:
        display_source_node(res_node, source_length=200)
        
plot_images(retrieved_images)



image_documents = [ImageDocument(image_path=image_path) for image_path in retrieved_images]


api_key = config("OPENROUTER_API_KEY")
base_url = config("OPENROUTER_BASEURL")
client = OpenAI(base_url=base_url, api_key=api_key)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": "Compare Qwen2.5 with Qwen3 using these images:"},
            
        ] + [
            {"type": "image_url",
             "image_url": {
                 "url": f"data:image/png;base64, {base64.b64encode(open(img, 'rb').read()).decode()}"
             },
             }
            for img in retrieved_images
        ]
    }
]

response = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
    max_tokens=512,
    )

print(response.choices[0].message.content)



