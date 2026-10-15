
from pathlib import Path

import pymupdf  # PyMuPDF


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_document(file_path: str | Path) -> list[dict]:
    """
    Load a TXT or PDF document.

    Returns one record for a TXT file and one record per
    non-empty PDF page.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            "Supported formats: TXT, PDF."
        )

    records = []

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8").strip()

        if text:
            records.append(
                {
                    "source": path.name,
                    "page": None,
                    "text": text,
                }
            )

    elif suffix == ".pdf":
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    records.append(
                        {
                            "source": path.name,
                            "page": page_number,
                            "text": text,
                        }
                    )

    return records


if __name__ == "__main__":
    documents_dir = Path("data/documents")

    for file_path in sorted(documents_dir.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        records = load_document(file_path)

        print(f"\nFile: {file_path.name}")
        print(f"Records extracted: {len(records)}")

        for record in records:
            print(
                f"Page: {record['page']} | "
                f"Characters: {len(record['text'])}"
            )
            print(record["text"][:300])
