import pymupdf as fitz
from typing import Dict, Any, List


class PDFParser:
    """Extracts text from PDF resumes while preserving layout order and detecting ATS traps."""

    @staticmethod
    def extract_text_from_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return PDFParser._parse_document(doc)

    @staticmethod
    def extract_text_from_file(file_path: str) -> Dict[str, Any]:
        doc = fitz.open(file_path)
        return PDFParser._parse_document(doc)

    @staticmethod
    def _parse_document(doc: fitz.Document) -> Dict[str, Any]:
        full_text_blocks = []
        ats_warnings = []
        page_count = len(doc)

        total_images = 0
        total_tables_detected = 0

        for page_idx, page in enumerate(doc):
            # 1. Layout-aware block extraction: (x0, y0, x1, y1, text, block_no, block_type)
            # block_type == 0 is text, 1 is image
            blocks = page.get_text("blocks")
            # Sort blocks primarily by vertical position (y0), then horizontal (x0)
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
            images_on_page = len(page.get_images())
            total_images += images_on_page

            # Check for multiple columns: if blocks have overlapping y spans and distinct x spans
            columns_detected = PDFParser._detect_multi_columns(text_blocks)
            if columns_detected:
                ats_warnings.append(
                    f"Page {page_idx + 1}: Multi-column layout detected. Some legacy ATS parsers may misread reading order."
                )

            # Sort blocks to preserve column-aware reading order
            sorted_blocks = PDFParser._sort_blocks_reading_order(text_blocks)
            for b in sorted_blocks:
                full_text_blocks.append(b[4].strip())

        raw_text = "\n\n".join(full_text_blocks)

        # ATS Warnings heuristics
        if total_images > 3:
            ats_warnings.append(
                f"Contains {total_images} embedded images/icons. Complex graphics can confuse parsing engines."
            )
        if len(raw_text.strip()) < 100:
            ats_warnings.append(
                "Extracted text is very short (<100 characters). This document might be an un-OCRed scanned image."
            )

        return {
            "text": raw_text,
            "page_count": page_count,
            "ats_warnings": ats_warnings,
            "has_scanned_appearance": len(raw_text.strip()) < 100 and total_images > 0
        }

    @staticmethod
    def _detect_multi_columns(blocks: List) -> bool:
        """Heuristic check: if there are distinct clusters of x0 coordinates."""
        if len(blocks) < 4:
            return False
        x0_coords = [b[0] for b in blocks]
        min_x = min(x0_coords)
        max_x = max(x0_coords)
        # If there's a spread across page width with multiple items starting far from the left margin
        mid_column_count = sum(1 for x in x0_coords if x > min_x + 150)
        return mid_column_count >= 3

    @staticmethod
    def _sort_blocks_reading_order(blocks: List) -> List:
        """Sort blocks: group into columns if separated by a significant x gap, otherwise by y0."""
        # Simple column split: check if page is partitioned into 2 columns
        if not blocks:
            return blocks
        left_blocks = [b for b in blocks if b[0] < 280]
        right_blocks = [b for b in blocks if b[0] >= 280]

        # If both left and right have substantial text, sort left column first, then right column
        if len(left_blocks) >= 3 and len(right_blocks) >= 3:
            left_sorted = sorted(left_blocks, key=lambda b: (b[1], b[0]))
            right_sorted = sorted(right_blocks, key=lambda b: (b[1], b[0]))
            return left_sorted + right_sorted

        # Otherwise standard top-to-bottom reading order
        return sorted(blocks, key=lambda b: (b[1], b[0]))
