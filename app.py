# app.py
import gradio as gr
from sentiment import analyze_sentiment
from summarization import summarize_text
from translation import translate_en_to_bn, translate_bn_to_en

def sentiment_interface(text):
    return analyze_sentiment(text)

def summarization_interface(text):
    return summarize_text(text)

def translation_interface(text, lang="en_to_bn"):
    if lang == "en_to_bn":
        return translate_en_to_bn(text)
    else:
        return translate_bn_to_en(text)

# Gradio Interface
iface = gr.Interface(
    fn=sentiment_interface, 
    inputs="text", 
    outputs="json", 
    live=True,
    title="BNLP Toolkit: Sentiment Analysis"
)

iface.add_tab("Summarization", gr.Interface(fn=summarization_interface, inputs="text", outputs="text"))
iface.add_tab("Translation", gr.Interface(fn=translation_interface, inputs=["text", gr.inputs.Radio(["en_to_bn", "bn_to_en"])], outputs="text"))

iface.launch()
