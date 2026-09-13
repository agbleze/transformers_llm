#%%
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


#%%
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from qdrant_client import QdrantClient, models


#%%

import shutil
shutil.rmtree("qdrant_index", ignore_errors=True)
print("Database cleared successfully!")

#%%

api_key = config("OPENROUTER_API_KEY")
base_url = config("OPENROUTER_BASEURL")


#%%

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%

pdf_url = "https://arxiv.org/pdf/2505.09388.pdf"
pdf_filename = "Qwen3.pdf"

"""

subprocess.run(["wget", "--user-agent", "Mozilla",
                pdf_url, "-O", pdf_filename
                ], check=True
               )

"""

#%%
os.path.exists(pdf_filename)


PDF_PATH = "Qwen3.pdf"

uploaded_pdf_path = Path(PDF_PATH)
output_dir = uploaded_pdf_path.stem

output_path = Path(f"{output_dir}")
output_path.mkdir(parents=True, exist_ok=True)

#%%

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

#%%
def extract_page_number(path: Union[str, Path]) -> Union[int, float]:
    if isinstance(path, ImageDocument):
        path = path.image_path or path.metadata.get("file_path")
    path = Path(path)
    match = re.search(f"page_(\d+)\.png", path.name)
    return int(match.group(1) if match else float("inf"))


image_paths_sorted_numeric = sorted(image_paths_sorted, key=extract_page_number)

#%%
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



#%% Plot 2 sample pages side by side at larger scale
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

#%% configure local text embedding model

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=device
)

#%% configure local image embedding model

image_embed_model = ClipEmbedding(model_name="ViT-B/32",
                                  device=device
                                  )


#%% loads the image files into ImageDocument objects for multimodal indexing

document_images = SimpleDirectoryReader("./Qwen3/").load_data()

#%%


#%%

len(document_images)

document_images[0].metadata.get("file_name")
#%%

client = qdrant_client.QdrantClient(path="qdrant_index")

if not client.collection_exists("text_collection"):
    client.create_collection(
        collection_name="text_collection",
        vectors_config=models.VectorParams(
            size=384,  # Dimension size for BAAI/bge-small-en-v1.5
            distance=models.Distance.COSINE
        )
    )
    print("Created missing text_collection.")

# 3. Check and manually create "image_collection" if missing
if not client.collection_exists("image_collection"):
    client.create_collection(
        collection_name="image_collection",
        vectors_config=models.VectorParams(
            size=512,  # Dimension size for CLIP ViT-B/32
            distance=models.Distance.COSINE
        )
    )
    print("Created missing image_collection.")

text_store = QdrantVectorStore(client=client,
                                collection_name="text_collection"
                                )

image_store = QdrantVectorStore(client=client, collection_name="image_collection")
storage_context = StorageContext.from_defaults(vector_store=text_store,
                                               image_store=image_store
                                               )


#%%    
#client.get_collection("image_collection").points_count


#%%
index = MultiModalVectorStoreIndex.from_documents(document_images,
                                                  storage_context=storage_context,
                                                  image_embed_model=image_embed_model
                                                  )

#%%

retriever_engine = index.as_retriever(image_similarity_top_k=2)

#%%
query = "Compare Qwen2.5 and Qwen3."
assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)
retrieval_results = retriever_engine.text_to_image_retrieve(query)

#%%
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
    

#%%   
retrieved_images = []
for res_node in retrieval_results:
    node = res_node.node
    if isinstance(node, ImageNode):
        path = node.image_path or node.metadata.get("file_path")
        if path:
            retrieved_images.append(path)
    else:
        display_source_node(res_node, source_length=200)


#%%        
plot_images(retrieved_images)


#%%
image_documents = [ImageDocument(image_path=image_path) for image_path in retrieved_images]

#%%
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
            for img in retrieved_images#[:3]
        ]
    }
]

#%%

response = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
    max_tokens=3000,
    )

#%%
print(response.choices[0].message.content)

#%%

print(response)

#%% load and find table data
documents_images_v2 = SimpleDirectoryReader("./Qwen3").load_data()


image_path = documents_images_v2[15].image_path

image = Image.open(image_path).convert("RGB")
plt.figure(figsize=(16, 9))
plt.imshow(image)
plt.axis("off")
plt.show()

#%%
image_prompt = """
Please load the table data and output it in JSON format from the image.
Try your best to extract the table data from the image.
If you can't extract the table data, summarize the image instead.
"""
#%%
with open(image_path, "rb") as f:
    image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")


#%%
response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {"role": "user",
         "content": [
             {"type": "text", "text": image_prompt.strip()},
             {"type": "image_url",
              "image_url": {"url": f"data:image/png;base64,{image_b64}"},
              }
         ]}
        
    ],
    max_tokens=2500
    )

#%%
response.choices[0].finish_reason

#%%
print(response.choices[0].message.reasoning)

#%%  ################ Document table extraction and description   #############
documents_images_v2_sorted = sorted(documents_images_v2, key=extract_page_number)

N = 10
documents_subset = documents_images_v2_sorted[:N]

#%%
image_results = {}


for idx, img_doc in enumerate(documents_subset, start=1):
    print(f"Processing image {idx}/{N}: {img_doc.image_path}")
    
    try:
        with open(img_doc.image_path, "rb") as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
        messages = [
            {"role": "user",
             "content": [
                 {"type": "text", "text": image_prompt.strip()},
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                  }
             ]}
        ]
        
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            max_tokens=1500
        )
        
        result_text = response.choices[0].message.content
        image_results[img_doc.image_path] = result_text
        
    except Exception as e:
        print(f"Error on image {idx}/{N}: {img_doc.image_path}. Error: {e}")
        continue
print(f"\n Done processing {len(image_results)} out of {N} images.")

#%%
text_docs = [
    Document(text=str(image_results[image_path]),
             metadata={"image_path": image_path}
             )
    for image_path in image_results
]
    
#%% 
    
drant_client = qdrant_client.QdrantClient(path="qdrant_mm_db_Qwen3")

#%%
from llama_index.llms.openai_like import OpenAILike

#%%
Settings.llm = OpenAILike(
    model="openrouter/free",            
    api_key=api_key,                    
    api_base=base_url,
    max_tokens=3000,
    is_chat_model=True                  
)
#%%
llama_text_store = QdrantVectorStore(client=drant_client,
                                     collection_name="text_collection"
                                     )

storage_context = StorageContext.from_defaults(vector_store=llama_text_store)

index = VectorStoreIndex.from_documents(text_docs, storage_context=storage_context)

#%%
MAX_TOKENS = 50
retriever_engine = index.as_retriever(similarity_top_k=3)
retrieval_results = retriever_engine.retrieve("Compare Qwen2.5 with Qwen3")

#%%
retrieved_image = []
for res_node in retrieval_results:
    display_source_node(res_node, source_length=1000)

#%%
query_engine = index.as_query_engine()

#%%
query_response = query_engine.query("Compare Qwen2.5 with Qwen3")

#%%

query_response.response
#%%

query_response
#%%
"""
detect table boxes, crop them and save for further analysis
"""

#%%
class MaxResize(object):
    def __init__(self, max_size: int = 800):
        self.max_size = max_size
        
    def __call__(self, image: PILImage.Image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.size((int(round(scale * width)), int(round(scale * height))))
        return resized_image
    
#%%
detection_transform = transforms.Compose([MaxResize(800),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]
                                                               )
                                          ]
                                         )


structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection",
    revision="no_timm"
).to("cuda" if torch.cuda.is_available() else "cpu")

structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to("cuda" if torch.cuda.is_available() else "cpu")


def box_cxcywh_to_xyxy(x: Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height])
    return boxes


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
    ]
    objects = []
    
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if class_label != "no object":
            objects.append({"label": class_label,
                            "score": float(score),
                            "bbox": [float(elem) for elem in bbox],
                            }
                           )
    return objects


def detect_and_crop_save_table(file_path,
                               cropped_table_directory = "./table_images/"
                               ):
    image = PILImage.open(file_path)
    filename, _ = os.path.splitext(os.path.basename(file_path))
    os.makedirs(cropped_table_directory, exist_ok=True)
    pixel_values = detection_transform(image).unsequeeze(0).to(model.device)
    
    with torch.no_grad():
        outputs = model(pixel_values)
    id2label = model.config.id2label
    id2label[len(id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)
    
    print(f"number of tables detected {len(detected_tables)}")
    
    for idx, obj in enumerate(detected_tables):
        cropped_table = image.crop(obj["bbox"])
        cropped_table.save(os.path.join(cropped_table_directory, f"{filename}_{idx}.png"))
        
        
def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = PILImage.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 9:
                break
    plt.tight_layout()
    plt.show()
            
            
for file_path in retrieval_results:
    detect_and_crop_save_table(file_path)
    
    
image_documents = SimpleDirectoryReader("./table_images/").load_data()


#%% 
"""
We send all cropped tables in one request and ask GPT4o for a direct
comparison that cites specific rows or metrics
"""
model = "openrouter/free"
api_key = config("OPENROUTER_API_KEY")
base_url = config("OPENROUTER_BASEURL")
client = OpenAI(base_url=base_url, api_key=api_key)

prompt = "Compare Qwen2.5 with Qwen3"

messages = [
    {
        "role": "user",
        "content": (
            [{"type": "text",
              "text": prompt
              }
             ] +
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(open(img.image_path, "rb").read()).decode('utf-8')}"
                    }
                },
                for img in image_documents
            ]
        )
    }
]
    
response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=1000
)

print(response.choices[0].message.content)