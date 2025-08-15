# PaperTopicClassiifier
Finetuning a BERT-based Transformer using PyTorch to classify scientific papers.


A machine learning project that fine-tunes a Transformer model to classify the **topic of a scientific paper** from its title and abstract. The trained model is deployed as an **interactive Streamlit app** and published on **Hugging Face Hub** for easy access.

## 🌐 Try It Online
[![PaperClassifier](https://huggingface.co/spaces/Smomitya/PaperClassifier)]

## 🚀 Features
- **Transformer-based architecture** (BERT or similar) fine-tuned on labeled scientific papers  
- **Input**: Paper title + abstract  
- **Output**: Predicted topic (e.g., Physics, Biology, Computer Science, etc.)  
- **Interactive UI** built with Streamlit  
- **Hugging Face Hub** integration for model hosting and sharing  

## 🛠 Tech Stack
- **Python**, **PyTorch**, **Transformers** (HuggingFace)  
- **Streamlit** for UI deployment  
- **Hugging Face Hub** for model hosting  

## 📊 Training
The model was fine-tuned on a **large dataset of arXiv papers**, containing thousands of samples across multiple scientific disciplines.

- **Data source**: arXiv API (titles, abstracts, categories)  
- **Preprocessing**: Text cleaning, category normalization, and tokenization using Hugging Face `AutoTokenizer`  
- **Model**: Pretrained BERT-based Transformer (`bert-base-uncased`)  
- **Training details**:  
  - Optimizer: AdamW  
  - Learning rate scheduler with warmup steps  
  - 3–5 epochs on GPU (NVIDIA Tesla T4)  
  - Stratified train/validation split  


