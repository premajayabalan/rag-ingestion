
from pathlib import Path

from docling.document_converter import DocumentConverter


# Supported document formats
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".csv",
}


def extract_document(file_path: str) -> str:
    """
    Extract structured content from a supported document
    and return it as Markdown.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    converter = DocumentConverter()
    result = converter.convert(str(path))

    markdown_content = result.document.export_to_markdown()

    return markdown_content


if __name__ == "__main__":
    input_file = "input/sample.docx"

    output_dir = Path("output/extracted")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(input_file).stem}.md"

    extracted_text = extract_document(input_file)

    output_file.write_text(
        extracted_text,
        encoding="utf-8",
    )

    print(f"Extraction complete: {output_file}")
    print(f"Extracted characters: {len(extracted_text)}")