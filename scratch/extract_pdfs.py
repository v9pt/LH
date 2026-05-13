import PyPDF2
import os

def extract_text(pdf_path, output_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += reader.pages[page_num].extract_text()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

files = [
    "GRP39_Up_Report.pdf",
    "IDS_BTP (2).pdf",
    "Annexure -I-BTP project report format (2) (1).pdf",
    "Annexure -II-BTP project report format (1) (1).pdf",
    "Notice-1 BTP project report format (1) (1) (1).pdf"
]

os.makedirs("scratch_docs", exist_ok=True)

for file in files:
    try:
        print(f"Extracting {file}...")
        extract_text(file, f"scratch_docs/{file.replace(' ', '_')}.txt")
    except Exception as e:
        print(f"Error extracting {file}: {e}")
