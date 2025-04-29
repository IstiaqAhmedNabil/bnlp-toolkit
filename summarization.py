# summarization.py
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("bengali"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 2)  # 2 sentences summary
    return ' '.join(str(sentence) for sentence in summary)
