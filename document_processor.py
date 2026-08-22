"""
document_processor.py - Academic Syllabus Document Ingestion and Chunking Engine.
Handles deterministic SHA-256 document hashing, page-aware text extraction,
hierarchical syllabus unit/section detection, and metadata-rich chunking.
"""

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import fitz  # PyMuPDF


@dataclass
class SyllabusChunk:
    """Represents a discrete semantic chunk of a syllabus document."""
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    unit: str
    section: str
    char_count: int

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk into Pinecone metadata dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": int(self.page),
            "unit": self.unit or "General",
            "section": self.section or "Overview",
            "text": self.text,
            "char_count": int(self.char_count)
        }


def compute_pdf_hash(pdf_bytes: bytes) -> str:
    """Calculate deterministic SHA-256 checksum for PDF content."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def extract_pages_from_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extract text per page with validation and metadata preservation.
    Returns:
        List of dicts: [{"page": 1, "text": "...", "char_count": 120}, ...]
    Raises:
        ValueError: If PDF is corrupt or contains zero extractable text.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{filename}': Corrupted or invalid format. Details: {e}")

    if len(doc) == 0:
        raise ValueError(f"PDF '{filename}' is empty (0 pages).")

    pages_data = []
    total_chars = 0

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        # Clean multiple spaces and irregular blank lines
        cleaned_text = re.sub(r"\r\n|\r", "\n", raw_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
        
        char_count = len(cleaned_text)
        total_chars += char_count
        
        if char_count > 0:
            pages_data.append({
                "page": page_num,
                "text": cleaned_text,
                "char_count": char_count
            })

    if total_chars < 30:
        raise ValueError(
            f"PDF '{filename}' contains little or no extractable text ({total_chars} characters found). "
            "It might be a scanned image or protected document."
        )

    return pages_data


# Regex patterns to detect academic units, modules, chapters, and sections
UNIT_PATTERN = re.compile(
    r"(?:^|\n)\s*(UNIT\s*[-–—:]*\s*(?:[I|V|X\d]+|\b(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b)|"
    r"MODULE\s*[-–—:]*\s*(?:[I|V|X\d]+|\b(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b)|"
    r"CHAPTER\s*[-–—:]*\s*\d+|"
    r"PART\s*[-–—:]*\s*(?:[I|V|X\d]+|[A-E]))",
    re.IGNORECASE
)

SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(\d+\.\d+\s+[A-Za-z0-9\s,–\-—/]{3,60}|"
    r"(?:Course Objectives|Course Outcomes|Prerequisites|Evaluation Scheme|"
    r"Reference Books|Text Books|Grading Policy|Weekly Schedule|Lab Experiments|Syllabus))",
    re.IGNORECASE
)


def detect_unit_and_section(text: str, current_unit: str = "General", current_section: str = "Overview") -> Tuple[str, str]:
    """
    Detect academic unit / module headers and sections in text snippets.
    Updates the running context as the parser encounters new headings.
    """
    unit_match = UNIT_PATTERN.search(text)
    if unit_match:
        detected_unit = unit_match.group(1).strip()
        # Find next line as potential unit topic title
        lines = [line.strip() for line in text[unit_match.end():].split("\n") if line.strip()]
        if lines and len(lines[0]) < 80:
            current_unit = f"{detected_unit}: {lines[0]}"
        else:
            current_unit = detected_unit

    sec_match = SECTION_PATTERN.search(text)
    if sec_match:
        current_section = sec_match.group(1).strip()

    return current_unit, current_section


def recursive_split_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Hierarchical recursive text splitter that prioritizes splitting along:
    1. Paragraph boundaries (\n\n)
    2. Academic list / unit boundaries (\n)
    3. Sentence terminators (. , ; : ! ?)
    4. Word boundaries (space)
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    # Find the highest level separator that exists in the text
    chosen_sep = " "
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for split in splits:
        piece = split if chosen_sep == "\n\n" or chosen_sep == "\n" else (split + chosen_sep)
        piece_len = len(piece)

        if current_len + piece_len > chunk_size and current_chunk:
            combined = "".join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Implement overlapping by keeping trailing pieces that fit within chunk_overlap
            overlap_pieces: List[str] = []
            overlap_len = 0
            for prev_piece in reversed(current_chunk):
                if overlap_len + len(prev_piece) <= chunk_overlap:
                    overlap_pieces.insert(0, prev_piece)
                    overlap_len += len(prev_piece)
                else:
                    break

            current_chunk = overlap_pieces
            current_len = overlap_len

        current_chunk.append(piece)
        current_len += piece_len

    if current_chunk:
        remaining = "".join(current_chunk).strip()
        if remaining:
            chunks.append(remaining)

    # Secondary pass: if any chunk is still larger than chunk_size, split with finer separators
    final_chunks: List[str] = []
    next_separators = separators[1:] if len(separators) > 1 else [" "]
    for chunk in chunks:
        if len(chunk) > chunk_size + chunk_overlap:
            sub_chunks = recursive_split_text(chunk, chunk_size, chunk_overlap, next_separators)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks


def process_pdf_into_chunks(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
    min_chunk_length: int = 60
) -> Tuple[str, List[SyllabusChunk]]:
    """
    Full pipeline to ingest a PDF and produce structured, page-aware syllabus chunks.
    
    Returns:
        (document_id, list_of_chunks)
    """
    document_id = compute_pdf_hash(pdf_bytes)
    pages_data = extract_pages_from_pdf(pdf_bytes, filename=filename)

    running_unit = "General"
    running_section = "Overview"
    all_chunks: List[SyllabusChunk] = []
    global_chunk_idx = 0

    for page_info in pages_data:
        page_num = page_info["page"]
        page_text = page_info["text"]

        # Track syllabus units across pages
        running_unit, running_section = detect_unit_and_section(
            page_text, running_unit, running_section
        )

        raw_splits = recursive_split_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for split in raw_splits:
            cleaned_split = split.strip()
            if len(cleaned_split) < min_chunk_length:
                continue

            # Update unit/section if specific split contains subheadings
            split_unit, split_section = detect_unit_and_section(
                cleaned_split, running_unit, running_section
            )
            if split_unit != running_unit:
                running_unit = split_unit
            if split_section != running_section:
                running_section = split_section

            # Deterministic unique chunk ID
            chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"

            # Context enrichment: Prefix each chunk with academic breadcrumbs
            enriched_text = (
                f"[{filename} | Page {page_num} | {running_unit} | {running_section}]\n"
                f"{cleaned_split}"
            )

            chunk = SyllabusChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                filename=filename,
                page=page_num,
                text=enriched_text,
                unit=running_unit,
                section=running_section,
                char_count=len(enriched_text)
            )
            all_chunks.append(chunk)
            global_chunk_idx += 1

    return document_id, all_chunks
