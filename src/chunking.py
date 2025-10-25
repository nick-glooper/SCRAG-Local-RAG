"""
Semantic chunking system for SecureRAG.
Intelligently splits documents into meaningful chunks.
"""

import re
import logging
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Intelligently chunk documents using semantic strategies.
    """

    def __init__(self, config):
        """
        Initialize chunker with configuration.

        Args:
            config: SecureRAGConfig instance or chunking config
        """
        # Extract chunking config
        if hasattr(config, 'chunking'):
            self.config = config.chunking
        else:
            self.config = config

        self.min_chunk_size = self.config.min_chunk_size
        self.max_chunk_size = self.config.max_chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.include_context = self.config.include_context
        self.strategy = self.config.strategy

        logger.info(f"Initialized chunker: strategy={self.strategy}, "
                   f"size={self.min_chunk_size}-{self.max_chunk_size}, "
                   f"overlap={self.chunk_overlap}")

    def chunk_document(
        self,
        text: str,
        metadata: Dict,
        filename: Optional[str] = None
    ) -> List[Dict]:
        """
        Chunk document using configured strategy.

        Args:
            text: Full document text
            metadata: Document metadata
            filename: Optional filename

        Returns:
            List of chunk dicts with text, raw_text, metadata, positions
        """
        if self.strategy == "semantic":
            return self._chunk_semantic(text, metadata, filename)
        elif self.strategy == "fixed":
            return self._chunk_fixed(text, metadata, filename)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using semantic")
            return self._chunk_semantic(text, metadata, filename)

    def _chunk_semantic(
        self,
        text: str,
        metadata: Dict,
        filename: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic chunking that respects document structure.
        """
        # Detect document structure
        sections = self._detect_sections(text)

        # If no clear structure, use hierarchical splitting
        if len(sections) <= 1:
            return self._chunk_hierarchical(text, metadata, filename)

        # Chunk each section
        all_chunks = []
        for section in sections:
            section_chunks = self._chunk_section(
                section["text"],
                section["header"],
                metadata,
                filename
            )
            all_chunks.extend(section_chunks)

        return all_chunks

    def _detect_sections(self, text: str) -> List[Dict]:
        """
        Detect document sections based on headers and structure.
        """
        sections = []

        # Try markdown-style headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')

        current_section = {
            "header": "",
            "text": "",
            "level": 0
        }

        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # Save previous section if it has content
                if current_section["text"].strip():
                    sections.append(current_section)

                # Start new section
                level = len(match.group(1))
                header = match.group(2)
                current_section = {
                    "header": header,
                    "text": line + "\n",
                    "level": level
                }
            else:
                current_section["text"] += line + "\n"

        # Add last section
        if current_section["text"].strip():
            sections.append(current_section)

        # If no markdown headers, try numbered sections
        if len(sections) <= 1:
            sections = self._detect_numbered_sections(text)

        return sections if len(sections) > 1 else [{"header": "", "text": text, "level": 0}]

    def _detect_numbered_sections(self, text: str) -> List[Dict]:
        """
        Detect numbered sections (1., 1.1, Article 1, etc.)
        """
        sections = []

        # Pattern for numbered sections
        patterns = [
            r'^(\d+\.(?:\d+\.)*)\s+(.+)$',  # 1.1, 1.1.1
            r'^(Article\s+\d+[:.]\s*)(.+)$',  # Article 1:
            r'^(Section\s+\d+[:.]\s*)(.+)$',  # Section 1:
        ]

        lines = text.split('\n')
        current_section = {"header": "", "text": "", "level": 0}

        for line in lines:
            is_header = False

            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section
                    if current_section["text"].strip():
                        sections.append(current_section)

                    # Start new section
                    current_section = {
                        "header": line.strip(),
                        "text": line + "\n",
                        "level": 1
                    }
                    is_header = True
                    break

            if not is_header:
                current_section["text"] += line + "\n"

        # Add last section
        if current_section["text"].strip():
            sections.append(current_section)

        return sections

    def _chunk_section(
        self,
        text: str,
        header: str,
        metadata: Dict,
        filename: Optional[str] = None
    ) -> List[Dict]:
        """
        Chunk a single section.
        """
        # If section is small enough, return as single chunk
        if len(text) <= self.max_chunk_size:
            chunk_text = self._add_context(text, header, metadata, filename)
            return [{
                "text": chunk_text,
                "raw_text": text,
                "metadata": {
                    **metadata,
                    "section": header
                },
                "char_start": 0,
                "char_end": len(text)
            }]

        # Otherwise, split hierarchically
        chunks = self._chunk_hierarchical(text, metadata, filename, header)
        return chunks

    def _chunk_hierarchical(
        self,
        text: str,
        metadata: Dict,
        filename: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[Dict]:
        """
        Hierarchical chunking: paragraphs → sentences → characters.
        """
        # Use LangChain's recursive splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                "! ",
                "? ",
                "; ",
                ": ",
                ", ",
                " ",     # Words
                ""       # Characters
            ]
        )

        # Split text
        text_chunks = splitter.split_text(text)

        # Create chunk objects
        chunks = []
        char_pos = 0

        for i, chunk_text in enumerate(text_chunks):
            # Skip tiny chunks
            if len(chunk_text) < self.min_chunk_size and i < len(text_chunks) - 1:
                continue

            # Find actual position in original text
            start_pos = text.find(chunk_text, char_pos)
            if start_pos == -1:
                start_pos = char_pos
            end_pos = start_pos + len(chunk_text)

            # Add context if enabled
            final_text = self._add_context(
                chunk_text,
                section,
                metadata,
                filename
            )

            chunk = {
                "text": final_text,
                "raw_text": chunk_text,
                "metadata": {**metadata},
                "char_start": start_pos,
                "char_end": end_pos
            }

            if section:
                chunk["metadata"]["section"] = section

            chunks.append(chunk)
            char_pos = end_pos

        return chunks

    def _chunk_fixed(
        self,
        text: str,
        metadata: Dict,
        filename: Optional[str] = None
    ) -> List[Dict]:
        """
        Simple fixed-size chunking with overlap.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for sep in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.min_chunk_size:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        end = start + len(chunk_text)
                        break

            # Add context
            final_text = self._add_context(chunk_text, None, metadata, filename)

            chunk = {
                "text": final_text,
                "raw_text": chunk_text,
                "metadata": {**metadata},
                "char_start": start,
                "char_end": end
            }

            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _add_context(
        self,
        chunk_text: str,
        section: Optional[str],
        metadata: Dict,
        filename: Optional[str]
    ) -> str:
        """
        Add contextual information to chunk text.
        """
        if not self.include_context:
            return chunk_text

        context_parts = []

        # Add filename
        if filename:
            context_parts.append(f"Document: {filename}")

        # Add section
        if section:
            context_parts.append(f"Section: {section}")

        # Add other relevant metadata
        if metadata.get("page"):
            context_parts.append(f"Page: {metadata['page']}")

        if not context_parts:
            return chunk_text

        context = "\n".join(context_parts)
        return f"{context}\n\n{chunk_text}"


def chunk_document(
    text: str,
    metadata: Dict,
    config: Dict,
    filename: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function to chunk a document.

    Args:
        text: Document text
        metadata: Document metadata
        config: Chunking configuration
        filename: Optional filename

    Returns:
        List of chunks
    """
    chunker = SemanticChunker(config)
    return chunker.chunk_document(text, metadata, filename)


if __name__ == "__main__":
    # Test chunking
    from src.config import load_config

    config = load_config()
    chunker = SemanticChunker(config)

    # Test text with sections
    test_text = """
# Introduction

This is the introduction to our document. It contains important information
about the topic we're discussing.

## Background

The background section provides context. It explains the history and
previous work in this area.

## Methodology

Our methodology involves several steps:
1. Data collection
2. Data processing
3. Analysis

# Results

The results show interesting findings. We discovered several key insights
that are relevant to our research question.

## Discussion

The discussion interprets our results in the broader context.
"""

    chunks = chunker.chunk_document(
        test_text,
        metadata={"source": "test"},
        filename="test_document.md"
    )

    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk['raw_text'])} chars")
        print(f"Section: {chunk['metadata'].get('section', 'N/A')}")
        print(f"Text preview: {chunk['text'][:200]}...")
