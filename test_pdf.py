from pypdf import PdfReader
import os

folder_path = "documents"

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, file)

        reader = PdfReader(pdf_path)

        print(f"\nReading: {file}")

        for page in reader.pages:
            text = page.extract_text()

            if text:
                print(text[:500])