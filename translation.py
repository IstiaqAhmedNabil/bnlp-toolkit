from transformers import MarianMTModel, MarianTokenizer

# Load the translation model for Bangla to English (BN→EN)
model_name = "Helsinki-NLP/opus-mt-bn-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_bn_to_en(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
