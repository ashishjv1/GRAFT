"""
Utility functions for context trimming
"""

import hashlib
from typing import Dict, Optional, Any, List
import numpy as np
import logging
from collections import OrderedDict

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Install with: pip install tiktoken")


class TokenCounter:
    """
    Token counting utility supporting multiple tokenizer models.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.

        Args:
            model_name: Model name for tokenizer (e.g., 'gpt-3.5-turbo', 'gpt-4')
        """
        self.model_name = model_name
        self.tokenizer = None

        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logging.warning(f"Unknown model {model_name}, using cl100k_base encoding")
        else:
            logging.warning("Using approximate token counting (tiktoken not available)")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: ~4 characters per token for English
            return max(1, len(text) // 4)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to specified token limit.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        if not text or max_tokens <= 0:
            return ""

        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        else:
            # Rough approximation
            max_chars = max_tokens * 4
            return text[:max_chars] if len(text) > max_chars else text


class EmbeddingCache:
    """
    LRU cache for storing text embeddings to avoid recomputation.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.

        Args:
            text: Input text

        Returns:
            Cached embedding or None if not found
        """
        key = self._hash_text(text)
        if key in self.cache:
            # Move to end (most recently used)
            embedding = self.cache.pop(key)
            self.cache[key] = embedding
            self.hits += 1
            return embedding.copy()  # Return copy to prevent modification

        self.misses += 1
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.

        Args:
            text: Input text
            embedding: Text embedding
        """
        key = self._hash_text(text)

        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = embedding.copy()

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size
        }


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end in the last 100 characters
            search_start = max(start, end - 100)
            sentence_end = -1

            for punct in ['. ', '! ', '? ', '\n\n']:
                pos = text.rfind(punct, search_start, end)
                if pos > sentence_end:
                    sentence_end = pos + len(punct)

            if sentence_end > start:
                end = sentence_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position (with overlap)
        start = end - overlap
        if start >= len(text):
            break

    return chunks


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity matrix for embeddings.

    Args:
        embeddings: Array of embeddings (n_samples, embedding_dim)

    Returns:
        Similarity matrix (n_samples, n_samples)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Calculate similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)

    return similarity_matrix


def diversify_selection(
    indices: List[int],
    embeddings: np.ndarray,
    diversity_threshold: float = 0.8,
    max_iterations: int = 10
) -> List[int]:
    """
    Remove highly similar items from selection to increase diversity.

    Args:
        indices: Selected indices
        embeddings: Full embedding matrix
        diversity_threshold: Similarity threshold for removal
        max_iterations: Maximum number of iterations

    Returns:
        Diversified selection indices
    """
    if len(indices) <= 1:
        return indices

    selected_embeddings = embeddings[indices]
    similarity_matrix = calculate_similarity_matrix(selected_embeddings)

    # Remove one item from each highly similar pair
    to_remove = set()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        found_similar = False

        for i in range(len(indices)):
            if i in to_remove:
                continue

            for j in range(i + 1, len(indices)):
                if j in to_remove:
                    continue

                if similarity_matrix[i, j] > diversity_threshold:
                    # Remove the one with lower average similarity to query
                    # (assuming query similarity is stored elsewhere, remove randomly for now)
                    to_remove.add(j)
                    found_similar = True
                    break

            if found_similar:
                break

        if not found_similar:
            break

    # Return indices not marked for removal
    final_indices = [indices[i] for i in range(len(indices)) if i not in to_remove]
    return final_indices


def validate_context_chunks(chunks: List[str], min_length: int = 10) -> List[str]:
    """
    Validate and clean context chunks.

    Args:
        chunks: List of text chunks
        min_length: Minimum length for valid chunks

    Returns:
        List of valid chunks
    """
    valid_chunks = []

    for chunk in chunks:
        if not isinstance(chunk, str):
            continue

        chunk = chunk.strip()

        # Skip empty or very short chunks
        if len(chunk) < min_length:
            continue

        # Skip chunks that are mostly whitespace or special characters
        if len(chunk.replace(' ', '').replace('\n', '').replace('\t', '')) < min_length // 2:
            continue

        valid_chunks.append(chunk)

    return valid_chunks