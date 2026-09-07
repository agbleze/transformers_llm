
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

device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
pdf_url = "https://arxiv.org/pdf/2505.09388.pdf"
pdf_filename = "Qwen3.pdf"

subprocess.run(["wget", "--user-agent", "Mozilla",
                pdf_url, "-O", pdf_filename
                ], check=True
               )

os.path.exists(pdf_filename)