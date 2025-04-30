# app.py
import gradio as gr
from sentiment import analyze_sentiment
from summarization import summarize_text
from translation import translate_bn_to_en


def sentiment_interface(text):
    return analyze_sentiment(text)

def summarization_interface(text):
    return summarize_text(text)

def translation_interface(text, lang="bn_to_en"):
    if lang == "bn_to_en":
        return translate_bn_to_en(text)
    else:
        return "EN → BN translation is not available."


# Gradio Interface Setup
sentiment_tab = gr.Interface(
    fn=sentiment_interface, 
    inputs="text", 
    outputs="json", 
    live=True,
    title="Sentiment Analysis"
)

summarization_tab = gr.Interface(
    fn=summarization_interface, 
    inputs="text", 
    outputs="text", 
    title="Text Summarization"
)

translation_tab = gr.Interface(
    fn=translation_interface, 
    inputs=[
        gr.Textbox(label="Input Text"), 
        gr.Radio(["bn_to_en", "en_to_bn"], label="Translation Direction", value="bn_to_en")
    ],
    outputs="text",
    title="Translation"
)

# Combine tabs into a single interface
gr.TabbedInterface(
    interface_list=[sentiment_tab, summarization_tab, translation_tab],
    tab_names=["Sentiment", "Summarization", "Translation"]
).launch()
