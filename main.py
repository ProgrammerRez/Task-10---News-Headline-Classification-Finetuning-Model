import os
import torch
from transformers import pipeline
import gradio as gr

MODEL_PATH = os.getenv('news_classifer_model')  # update to your model dir or HF repo id
device = 0 if torch.cuda.is_available() else -1

clf = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, device=device, return_all_scores=True)

def classify_headline(text: str):
    if not text or not text.strip():
        return {}, {}
    outputs = clf(text)[0]
    probs = {o["label"]: float(o["score"]) for o in sorted(outputs, key=lambda x: x["score"], reverse=True)}
    return probs, probs

with gr.Blocks() as demo:
    gr.Markdown("## News Headline Classifier")
    inp = gr.Textbox(lines=2, placeholder="Enter a news headline...", label="Headline")
    lbl = gr.Label(num_top_classes=5, label="Predicted")
    out_json = gr.JSON(label="All class probabilities")
    btn = gr.Button("Classify")
    btn.click(classify_headline, inputs=inp, outputs=[lbl, out_json])

if __name__ == "__main__":
    demo.launch(server_port=int(8000),share=True)
