import re
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from transformers import AutoTokenizer, TextStreamer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
import time
import requests
import gradio as gr
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
