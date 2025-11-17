from typing import List, Protocol
from pydantic import Field
import re
from llama_index.core.node_parser import SentenceSplitter

class Splitter(Protocol):
    """Splitter 协议：任何 splitter 实现都需提供 split_text 方法"""
    def split_text(self, text: str) -> List[str]:
        ...

class SentenceSplitterWrapper:
    """SentenceSplitterWrapper using llama_index#SentenceSplitter underlying"""
    def __init__(self, *args, **kwargs):
        self._splitter = SentenceSplitter(args, kwargs)

    def split_text(self, text: str) -> List[str]:
        return self._splitter.split_text(text)
    
class ChineseTextSplitter:
    """
    Parse text with a preference for complete sentences. Specifically working for chinese or mixed-chinese text.
    """

    DEFAULT_CHUNK_SIZE=800
    SENTENCE_CHUNK_OVERLAP=20
    DEFAULT_SEPERATORS="。|；|!|?|\n"
    DEFAULT_PARAGRAPH_SEP="\n\n"

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The character chunk size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=SENTENCE_CHUNK_OVERLAP,
        description="The character overlap of each chunk when splitting.",
        ge=0,
    )
    separator: str = Field(
        default=DEFAULT_SEPERATORS, description="Default separator for splitting into sentences"
    )
    paragraph_separator: str = Field(
        default=DEFAULT_PARAGRAPH_SEP, description="Separator between paragraphs."
    )

    def __init__(
        self,
        separator: str = DEFAULT_SEPERATORS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEP,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator

    def split_text(self, text: str) -> List[str]:
        paragraphs = self._rebuild_paragraphs(text)
        chunk_data_list = []

        for para in paragraphs:
            # Split paragraph into basic chunks
            chunks = self._split_text(para)
            if not chunks:
                continue

            # Add overlap between consecutive chunks
            overlapped_chunks = []
            prev_chunk = ""

            for i, chunk in enumerate(chunks):
                if len(chunk) < self.chunk_overlap and len(prev_chunk) < self.chunk_overlap:
                    prev_chunk = +chunk
                    continue

                if i == 0:
                    # First chunk — no overlap
                    overlapped_chunks.append(chunk)
                else:
                    # Add last N chars from previous chunk to current one
                    overlap_part = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) >= self.chunk_overlap else prev_chunk
                    merged_chunk = overlap_part + chunk
                    overlapped_chunks.append(merged_chunk)

                prev_chunk = chunk  # Update for next round

            # Extend to global result
            chunk_data_list.extend(overlapped_chunks)
            if prev_chunk:
                chunk_data_list.append(prev_chunk)

        return chunk_data_list

    def _rebuild_paragraphs(self, text: str) -> List[str]:
        paragraphs = []
        current_para = []
        current_len = 0

        for para in text.split("\n\n"):
            para = para.strip()
            para_len = len(para)
            if para_len == 0:
                continue
            if current_len + para_len <= self.chunk_size:
                current_para.append(para)
                current_len += para_len
            else:
                if current_para:
                    paragraphs.append("\n".join(current_para))
                current_para = [para]
                current_len = para_len

        if current_para:
            paragraphs.append("\n".join(current_para))

        return paragraphs
    
    def _split_text(self, text: str) -> List[str]:
        splits = re.split(f'({self.separator})', text)
        chunks = []
        current_chunk = []
        for part in splits:
            part = part.strip()
            if not part:
                continue
            if re.fullmatch(self.separator, part):
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
            else:
                current_chunk.append(part)
        if current_chunk:
            chunks.append("".join(current_chunk))
        return [chunk.strip() for chunk in chunks if chunk.strip()]