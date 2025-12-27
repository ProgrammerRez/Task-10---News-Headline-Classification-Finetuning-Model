# 📰 News Headline Classifier (Transformers + Gradio)

A lightweight **news headline classification app** built using **Hugging Face Transformers**, **PyTorch**, and **Gradio**.

It loads a **pretrained text classification model** (local or from Hugging Face Hub) and exposes it through a simple web UI to classify news headlines and display **full class probability distributions**.

---

## 🚀 Features

* 🧠 Transformer-based **text classification**
* ⚡ GPU acceleration (automatically enabled if available)
* 📊 Displays **top predictions + full probability scores**
* 🖥️ Interactive **Gradio web interface**
* 🔁 Supports **local models or Hugging Face Hub models**

---

## 🧠 How It Works

1. A text classification pipeline is initialized using:

   * `MODEL_PATH` (local directory or HF repo ID)
2. The user enters a news headline
3. The model outputs:

   * Top predicted labels
   * Full probability distribution across all classes
4. Results are displayed using Gradio components

---

## 🧱 Tech Stack

* **Python 3.9+**
* **PyTorch**
* **Hugging Face Transformers**
* **Gradio**

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/news-headline-classifier.git
cd news-headline-classifier
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install torch transformers gradio
```

---

## 🔑 Model Setup

Set the model path or Hugging Face repo ID via an environment variable:

```bash
export news_classifer_model="your-model-path-or-hf-repo"
```

Windows (PowerShell):

```powershell
setx news_classifer_model "your-model-path-or-hf-repo"
```

Examples:

```text
distilbert-base-uncased-finetuned-sst-2-english
./models/news_classifier
```

---

## ▶️ Running the App

```bash
python app.py
```

The app will launch at:

```
http://localhost:8000
```

A public Gradio share link will also be generated.

---

## 🖥️ UI Components

* **Textbox** – enter a news headline
* **Label** – shows top predicted classes
* **JSON Viewer** – shows all class probabilities
* **Button** – triggers inference

---

## ⚠️ Important Notes

* The model **must be a text classification model**
* `return_all_scores=True` is enabled to expose full probabilities
* Empty or whitespace-only input returns empty outputs
* GPU is used automatically if CUDA is available

---

## 🧪 Example Use Cases

* News topic classification
* Headline moderation
* Dataset exploration
* Model demos and quick validation
* Lightweight ML inference apps

---

## 📌 Limitations

* Single-input inference (no batching)
* No confidence thresholding
* No preprocessing beyond tokenizer defaults
* Not optimized for high-throughput production use

---

## 📜 License

MIT License

---



This is a **clean demo-style deployment**, not a production system. That’s fine—as long as you know the difference.
