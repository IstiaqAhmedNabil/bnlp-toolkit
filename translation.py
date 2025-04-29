# translation.py
from huggingface_hub import login
from transformers import MarianMTModel, MarianTokenizer

# Manually authenticate with Hugging Face using your token
from huggingface_hub import login
import os

login(os.getenv("HF_TOKEN"))


# Load the translation model for English to Bangla (EN↔BN)
model_name = "Helsinki-NLP/opus-mt-en-bn"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_en_to_bn(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_bn_to_en(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
