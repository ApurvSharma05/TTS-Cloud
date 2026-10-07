"""
test_document_processor.py - Unit tests for syllabus document ingestion,
unit detection heuristics, hashing, and recursive text splitting.
"""

import pytest
from document_processor import (
    compute_pdf_hash,
    detect_unit_and_section,
    recursive_split_text,
    SyllabusChunk
)


def test_compute_pdf_hash():
    sample_bytes_1 = b"%PDF-1.4 sample content for syllabus"
    sample_bytes_2 = b"%PDF-1.4 different content"
    hash_1 = compute_pdf_hash(sample_bytes_1)
    hash_2 = compute_pdf_hash(sample_bytes_2)

    assert len(hash_1) == 16
    assert len(hash_2) == 16
    assert hash_1 != hash_2
    assert compute_pdf_hash(sample_bytes_1) == hash_1


def test_detect_unit_and_section_unit_detection():
    text_1 = "UNIT III: RELATIONAL DATABASE DESIGN\n3.1 Functional Dependencies"
    unit, sec = detect_unit_and_section(text_1)
    assert "UNIT III" in unit
    assert "RELATIONAL DATABASE DESIGN" in unit
    assert "3.1" in sec

    text_2 = "MODULE 4 - Process Scheduling & CPU Optimization\nCourse Outcomes:"
    unit, sec = detect_unit_and_section(text_2)
    assert "MODULE 4" in unit
    assert "Course Outcomes" in sec


def test_detect_unit_and_section_context_preservation():
    # When text does not contain new unit headers, it should preserve running unit
    text_snippet = "This section discusses Boyce-Codd Normal Form and multi-valued dependencies."
    unit, sec = detect_unit_and_section(
        text_snippet,
        current_unit="Unit 3: Normalization",
        current_section="3.3 BCNF"
    )
    assert unit == "Unit 3: Normalization"
    assert sec == "3.3 BCNF"


def test_recursive_split_text_respects_chunk_size():
    long_text = "Database normalization minimizes redundancy. " * 30  # ~1350 characters
    chunks = recursive_split_text(long_text, chunk_size=400, chunk_overlap=80)

    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should reasonably fit within chunk_size + overlap tolerance
        assert len(chunk) <= 480


def test_recursive_split_text_preserves_short_text():
    short_text = "Single short syllabus sentence."
    chunks = recursive_split_text(short_text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_syllabus_chunk_to_metadata():
    chunk = SyllabusChunk(
        chunk_id="doc1_p1_c0",
        document_id="doc1",
        filename="CS101.pdf",
        page=3,
        text="Sample syllabus text",
        unit="Unit 1",
        section="1.1 Overview",
        char_count=20
    )
    meta = chunk.to_metadata()
    assert meta["chunk_id"] == "doc1_p1_c0"
    assert meta["page"] == 3
    assert meta["unit"] == "Unit 1"
    assert meta["filename"] == "CS101.pdf"
