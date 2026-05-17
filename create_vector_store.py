from pypdf import PdfReader
import os
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Read PDFs and TXT files
# -----------------------------

folder_path = "documents"

documents = []

for file in os.listdir(folder_path):

    file_path = os.path.join(folder_path, file)

    text_content = ""

    # PDF files
    if file.endswith(".pdf"):

        reader = PdfReader(file_path)

        for page in reader.pages:

            text = page.extract_text()

            if text:

                text_content += text + "\n"

    # TXT transcript files
    elif file.endswith(".txt"):

        with open(file_path, "r", encoding="utf-8") as f:

            text_content += f.read() + "\n"

    # Store file + content
    if text_content.strip():

        documents.append({
            "source": file,
            "content": text_content
        })

# -----------------------------
# Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)

all_chunks = []
all_sources = []

for doc in documents:

    chunks = text_splitter.split_text(
        doc["content"]
    )

    for chunk in chunks:

        all_chunks.append(chunk)

        all_sources.append(doc["source"])

print(f"Total chunks created: {len(all_chunks)}")

# -----------------------------
# TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(all_chunks)

# -----------------------------
# Save everything
# -----------------------------

with open("vector_store.pkl", "wb") as f:

    pickle.dump(
        (
            vectorizer,
            vectors,
            all_chunks,
            all_sources
        ),
        f
    )

print("Offline TF-IDF vector store created successfully!")