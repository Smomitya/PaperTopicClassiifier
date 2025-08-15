import streamlit as st
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def top_proba(preds, p=.95):
    preds.sort(key=lambda pr: pr["score"])
    top_preds = []
    proba = 0.
    for pr in reversed(preds):
        proba += pr["score"]
        top_preds.append(pr)
        if proba >= p:
            return top_preds

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("Smomitya/distilbert-paper-classifier")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

model = load_model()

st.title("Topic Classification")

title = st.text_input("Enter the title of the article:")
title_only = st.checkbox("Use only title for classification")
if not title_only:
    abstract = st.text_input("Add the abstract or description of the article:")
st.header("Results")
show_scores = st.checkbox("Show scores")
if title and (title_only or abstract):
    query = title if title_only else title + " " + abstract
    preds = top_proba(model(query)[0])
    for pr in preds:
        label = pr["label"]
        score = pr["score"]
        st.markdown(f"`{label}` " + (f"{score:.3f}" if show_scores else ""))
