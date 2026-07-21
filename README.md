# -Unprof_Day19
Python Intermediate Day 19  Assignment
# Day 19 - RAG Architecture

## Overview

This project demonstrates the basic workflow of Retrieval-Augmented Generation (RAG).

## Workflow

1. Load documents
2. Generate embeddings
3. Store embeddings in FAISS
4. Convert user query into embedding
5. Retrieve similar document chunks
6. Generate response using retrieved context

## Technologies

- Python
- Sentence Transformers
- FAISS
- NumPy

## Project Structure

```
Day19-RAG-Architecture/
│── app.py
│── utils.py
│── requirements.txt
│── README.md
│── data/
│    └── knowledge.txt
```

## Run

```bash
pip install -r requirements.txt
python app.py
```
