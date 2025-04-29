# sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the pre-trained Bangla sentiment analysis model
model_name = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    return sentiment_analyzer(text)
