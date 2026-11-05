import re
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

from src.classification.classification_service import classify_chunk


MODEL_NAME = "microsoft/harrier-oss-v1-270M"


@dataclass
class Chunk:
    chunk_id: int
    document_id: Optional[str]
    chunk_index: int
    content: str
    section_path: str
    content_type: str
    token_count: int
    metadata: dict = field(default_factory=dict)


class StructureAwareChunker:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        model_name: str = MODEL_NAME,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Higher scores indicate preferred split boundaries.
        self.boundary_rules = [
            (r"(?m)(?=^#{1,6}\s+)", 100),  # Heading
            (r"\n\s*\n", 60),             # Paragraph
            (r"(?<=[.!?])\s+", 40),       # Sentence
            (r"\n", 30),                  # Line
            (r"\s+", 5),                  # Word
        ]

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def split_markdown_into_sections(self, markdown: str):
        """
        Split Markdown into heading-based sections while
        preserving the complete heading hierarchy.
        """
        sections = []
        heading_stack = []
        current_lines = []

        def save_section():
            content = "\n".join(current_lines).strip()

            if content:
                sections.append({
                    "content": content,
                    "section_path": " > ".join(heading_stack),
                })

        for line in markdown.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if match:
                save_section()
                current_lines.clear()

                level = len(match.group(1))
                heading = match.group(2).strip()

                heading_stack[:] = heading_stack[:level - 1]
                heading_stack.append(heading)

                current_lines.append(line)
            else:
                current_lines.append(line)

        save_section()
        return sections

    def _find_split_candidates(self, text: str):
        """
        Find candidate boundaries and assign their rule scores.
        """
        candidates = []

        for pattern, score in self.boundary_rules:
            for match in re.finditer(pattern, text):
                start = match.start()
                end = match.end()

                if start <= 0 or end >= len(text):
                    continue

                candidates.append({
                    "left_end": start,
                    "right_start": end,
                    "score": score,
                })

        return candidates

    def _choose_split(self, text: str, max_tokens: int):
        """
        Select a high-scoring boundary that fits within the
        maximum token limit.
        """
        candidates = self._find_split_candidates(text)
        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:
            left = text[:candidate["left_end"]].strip()
            right = text[candidate["right_start"]:].strip()

            if not left or not right:
                continue

            left_tokens = self.count_tokens(left)

            if left_tokens > max_tokens:
                continue

            # Prefer boundaries close to the token limit,
            # while giving structural boundaries more weight.
            closeness = left_tokens / max_tokens
            combined_score = (
                candidate["score"] + (25 * closeness)
            )

            if combined_score > best_score:
                best_score = combined_score
                best_candidate = (left, right)

        return best_candidate

    def recursive_split(self, text: str, max_tokens: int):
        """
        Recursively split oversized text using scored
        structural boundaries.
        """
        text = text.strip()

        if not text:
            return []

        if self.count_tokens(text) <= max_tokens:
            return [text]

        split = self._choose_split(text, max_tokens)

        if split is None:
            return self.split_by_tokens(text, max_tokens)

        left, right = split

        return (
            self.recursive_split(left, max_tokens)
            + self.recursive_split(right, max_tokens)
        )

    def split_by_tokens(self, text: str, max_tokens: int):
        """
        Last-resort token splitting when no suitable
        structural boundary exists.
        """
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        result = []

        for start in range(0, len(token_ids), max_tokens):
            window = token_ids[start:start + max_tokens]

            decoded = self.tokenizer.decode(
                window,
                skip_special_tokens=True,
            ).strip()

            if decoded:
                result.append(decoded)

        return result

    def detect_content_type(self, text: str) -> str:
        """
        Infer the dominant Markdown content type.
        """
        lines = text.splitlines()

        if any(
            line.strip().startswith("|")
            for line in lines
        ):
            return "table"

        if any(
            re.match(r"^\s*([-*+]|\d+\.)\s+", line)
            for line in lines
        ):
            return "list"

        return "text"

    def _pack_pieces(self, pieces, max_tokens):
        """
        Pack recursive pieces into chunks without exceeding
        the given token limit.
        """
        packed = []
        current = []

        for piece in pieces:
            candidate = "\n\n".join(current + [piece])

            if self.count_tokens(candidate) <= max_tokens:
                current.append(piece)
            else:
                if current:
                    packed.append("\n\n".join(current))

                current = [piece]

        if current:
            packed.append("\n\n".join(current))

        return packed

    def create_chunks(
        self,
        markdown: str,
        document_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ):
        sections = self.split_markdown_into_sections(markdown)

        chunks = []
        chunk_id = 0

        for section in sections:
            section_path = section["section_path"]
            content = section["content"]

            heading_prefix = (
                f"Section: {section_path}\n\n"
                if section_path
                else ""
            )

            prefix_tokens = self.count_tokens(heading_prefix)
            available_tokens = self.chunk_size - prefix_tokens

            if available_tokens <= 0:
                raise ValueError(
                    "Section heading exceeds the configured chunk size."
                )

            # Reserve space for overlap in every chunk.
            new_content_capacity = (
                available_tokens - self.chunk_overlap
            )

            if new_content_capacity <= 0:
                raise ValueError(
                    "Chunk size must exceed heading tokens "
                    "plus overlap tokens."
                )

            pieces = self.recursive_split(
                content,
                new_content_capacity,
            )

            raw_chunks = self._pack_pieces(
                pieces,
                new_content_capacity,
            )

            previous_token_ids = []

            for index, raw_chunk in enumerate(raw_chunks):
                current_token_ids = self.tokenizer.encode(
                    raw_chunk,
                    add_special_tokens=False,
                )

                if index > 0 and self.chunk_overlap > 0:
                    overlap_ids = previous_token_ids[
                        -self.chunk_overlap:
                    ]

                    current_token_ids = (
                        overlap_ids + current_token_ids
                    )

                # Defensive hard-limit enforcement.
                current_token_ids = current_token_ids[
                    :available_tokens
                ]

                chunk_body = self.tokenizer.decode(
                    current_token_ids,
                    skip_special_tokens=True,
                ).strip()

                chunk_content = heading_prefix + chunk_body
                token_count = self.count_tokens(chunk_content)

                if token_count > self.chunk_size:
                    raise RuntimeError(
                        "Chunk exceeded the configured token limit."
                    )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=len(chunks),
                        content=chunk_content,
                        section_path=section_path,
                        content_type=self.detect_content_type(
                            chunk_body
                        ),
                        token_count=token_count,
                        metadata={
                            "source_file": source_file,
                            "section_path": section_path,
                            "chunk_overlap": (
                                self.chunk_overlap if index > 0 else 0
                            ),
                        },
                    )
                )

                chunk_id += 1
                previous_token_ids = self.tokenizer.encode(
                    chunk_body,
                    add_special_tokens=False,
                )

        return chunks


if __name__ == "__main__":
    input_file = Path("output/extracted/sample.md")
    output_file = Path("output/chunks/chunks.json")

    markdown = input_file.read_text(encoding="utf-8")

    chunker = StructureAwareChunker(
        chunk_size=600,
        chunk_overlap=100,
    )

    chunks = chunker.create_chunks(
        markdown,
        document_id="sample-document-001",
        source_file=input_file.name,
    )

    # Classify each chunk and attach classification metadata.
    for chunk in chunks:
        result = classify_chunk(chunk.content)

        chunk.metadata.update({
            "classification": result["classification"].name,
            "classification_score": result["score"],
            "classification_reason": result["reason"],
        })

        print(
            f"Chunk {chunk.chunk_index}: "
            f"{result['classification'].name} "
            f"(score: {result['score']})"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(
            [asdict(chunk) for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Created {len(chunks)} chunks")
    print(f"Saved chunks to: {output_file}")