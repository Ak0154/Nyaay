# services/pdf_loader.py

import pdfplumber

from langchain_core.documents import Document


def extract_pdf_text(pdf_path):

    documents = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages):

            all_text = ""

            # ---------------- Normal Text ----------------

            text = page.extract_text()

            if text:

                all_text += (
                    f"\n--- Page {page_num + 1} ---\n"
                )

                all_text += text + "\n"

            # ---------------- Tables ----------------

            tables = page.extract_tables()

            for table_num, table in enumerate(
                tables
            ):

                all_text += (
                    f"\n[TABLE {table_num+1}] "
                    f"(Page {page_num+1})\n"
                )

                for row in table:

                    row_text = " | ".join(
                        str(cell)
                        if cell else ""
                        for cell in row
                    )

                    all_text += (
                        row_text + "\n"
                    )

            # ---------------- LangChain Document ----------------

            documents.append(

                Document(

                    page_content=all_text,

                    metadata={

                        "page": page_num + 1,
                        "source": str(pdf_path)

                    }

                )

            )

    return documents