# NFC Offline AI Assistant

An offline document retrieval assistant built using Python, Streamlit, and TF-IDF.

## Features

- PDF document processing
- TXT transcript processing
- Text chunking
- TF-IDF vectorization
- Cosine similarity retrieval
- Offline operation

## Installation

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

## Build Vector Store

python create_vector_store.py

## Run Application

streamlit run app.py
