# PaperTopicClassiifier
Finetuning a BERT-based Transformer using PyTorch to classify scientific papers.


A machine learning project that fine-tunes a Transformer model to classify the **topic of a scientific paper** according to [the specific taxonomy](https://arxiv.org/category_taxonomy) from its title and abstract. The trained model is deployed as an **interactive Streamlit app** and published on **Hugging Face Hub** for easy access.

## Try It Online
[PaperClassifier](https://huggingface.co/spaces/Smomitya/PaperClassifier)

## How It Works
1. **User Input**: The user enters a scientific paper title and/or abstract into the app.  
2. **Preprocessing**: The text is tokenized and converted into numerical embeddings using the pretrained Transformer tokenizer.  
3. **Prediction**: The fine-tuned classifier processes the embeddings and outputs probabilities for each possible scientific category.  
4. **Results**: The app displays a short ranked list of the **most likely categories**, sorted by prediction probability.


## Features
- **Transformer-based architecture** (BERT or similar) fine-tuned on labeled scientific papers  
- **Input**: Paper title + abstract  
- **Output**: Predicted topic (e.g., Physics, Biology, Computer Science, etc.)  
- **Interactive UI** built with Streamlit  
- **Hugging Face Hub** integration for model hosting and sharing  

## Tech Stack
- **Python**, **PyTorch**, **Transformers** (HuggingFace)  
- **Streamlit** for UI deployment  
- **Hugging Face Hub** for model hosting  

## Training
The model was fine-tuned on a **large dataset of arXiv papers**, containing thousands of samples across multiple scientific disciplines.
- **Data source**: [arXiv storage (titles, abstracts, categories)](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **Preprocessing**: Text cleaning, category normalization, and tokenization using Hugging Face `AutoTokenizer`  
- **Model**: Pretrained BERT-based Transformer (`bert-base-cased`)  
- **Training details**:  
  - Optimizer: AdamW  
  - Learning rate scheduler with warmup steps  
  - 5 epochs on GPU (NVIDIA A100 in the Cloud)  
  - Stratified train/validation split  


