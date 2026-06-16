from pypdf import PdfReader
import os
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

folder_path = "documents"

documents = []

# Read PDFs and TXT files
for file in os.listdir(folder_path):

    file_path = os.path.join(folder_path, file)

    if file.endswith(".pdf"):

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

        documents.append({
            "text": text,
            "source": file
        })

    elif file.endswith(".txt"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

            documents.append({
                "text": text,
                "source": file
            })

# Better chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=250
)

chunks = []
sources = []

for doc in documents:

    split_chunks = text_splitter.split_text(
        doc["text"]
    )

    for chunk in split_chunks:

        chunks.append(chunk)
        sources.append(doc["source"])

print(f"Total chunks created: {len(chunks)}")

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

vectors = vectorizer.fit_transform(chunks)

with open("vector_store.pkl", "wb") as f:

    pickle.dump(
        (
            vectorizer,
            vectors,
            chunks,
            sources
        ),
        f
    )

print("Offline TF-IDF vector store created successfully!")