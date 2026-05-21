import pdfplumber
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

pdf_folder = BASE_DIR / "dataset"
output_folder = BASE_DIR / "output"

os.makedirs(output_folder, exist_ok=True)

# Find PDFs regardless of extension case
pdf_files = [
    f for f in pdf_folder.iterdir()
    if f.is_file() and f.suffix.lower() == ".pdf"
]

print("Found files:")
print(pdf_files)

for pdf_file in pdf_files:

    print(f"Processing: {pdf_file.name}")

    all_text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page_num, page in enumerate(pdf.pages):

            # Extract text
            text = page.extract_text()

            if text:
                all_text += f"\n--- Page {page_num+1} ---\n"
                all_text += text + "\n"

            # Extract tables
            tables = page.extract_tables()

            for table_num, table in enumerate(tables):

                all_text += (
                    f"\n--- Table {table_num+1} "
                    f"(Page {page_num+1}) ---\n"
                )

                for row in table:
                    row_text = " | ".join(
                        str(cell) if cell else ""
                        for cell in row
                    )
                    all_text += row_text + "\n"

    output_path = output_folder / f"{pdf_file.stem}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"Saved: {output_path}")

print("Done.")