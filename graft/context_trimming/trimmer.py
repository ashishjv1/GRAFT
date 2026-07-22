"""
Core Context Trimming Implementation using GRAFT principles
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import itertools
import copy
from tqdm import tqdm
import logging

try:
    from sentence_transformers import SentenceTransformer
    import tiktoken
    OPTIONAL_DEPS = True
except ImportError:
    OPTIONAL_DEPS = False
    logging.warning("Optional dependencies not installed. Run: pip install sentence-transformers tiktoken")

from ..decompositions import index_sel
from ..grad_dist import calnorm
from .utils import TokenCounter, EmbeddingCache


class ContextTrimmer:
    """
    Intelligent context trimming using GRAFT's gradient-based importance scoring.
    Adapts the core GRAFT algorithm for text/prompt optimization under budget constraints.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        selection_fraction: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
        tokenizer_model: str = "gpt-3.5-turbo",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_size: int = 1000,
        batch_size: int = 32
    ):
        """
        Initialize the Context Trimmer.

        Args:
            max_tokens: Maximum number of tokens allowed in context
            selection_fraction: Fraction of content to keep (0.0 to 1.0)
            embedding_model: SentenceTransformer model for embeddings
            tokenizer_model: Tokenizer model for token counting
            device: Computing device (cuda/cpu)
            cache_size: Size of embedding cache
            batch_size: Batch size for processing
        """
        if not OPTIONAL_DEPS:
            raise ImportError("Required dependencies missing. Install with: pip install graft-pytorch[context]")

        self.max_tokens = max_tokens
        self.selection_fraction = selection_fraction
        self.device = device
        self.batch_size = batch_size

        # Initialize components
        self.embedding_model = SentenceTransformer(embedding_model)
        self.token_counter = TokenCounter(tokenizer_model)
        self.embedding_cache = EmbeddingCache(cache_size)

        logging.info(f"ContextTrimmer initialized - Max tokens: {max_tokens}, Selection: {selection_fraction}")

    def trim_context(
        self,
        context_chunks: List[str],
        query: str,
        importance_weights: Optional[List[float]] = None,
        preserve_order: bool = True
    ) -> Dict[str, Union[List[str], Dict]]:
        """
        Trim context using GRAFT-based importance scoring.

        Args:
            context_chunks: List of text chunks to potentially include
            query: The user query/prompt
            importance_weights: Optional manual importance weights per chunk
            preserve_order: Whether to maintain original chunk order

        Returns:
            Dictionary containing selected chunks and metadata
        """
        if not context_chunks:
            return {"selected_chunks": [], "metadata": {"total_tokens": 0}}

        # Count tokens for budget awareness
        chunk_tokens = [self.token_counter.count_tokens(chunk) for chunk in context_chunks]
        query_tokens = self.token_counter.count_tokens(query)

        total_available = self.max_tokens - query_tokens
        if total_available <= 0:
            logging.warning("Query exceeds max token limit")
            return {"selected_chunks": [], "metadata": {"total_tokens": query_tokens}}

        # Get embeddings with caching
        embeddings = self._get_embeddings_cached(context_chunks + [query])
        context_embeddings = embeddings[:-1]
        query_embedding = embeddings[-1:]

        # Perform GRAFT-based selection
        selected_indices = self._graft_selection(
            context_embeddings,
            query_embedding,
            chunk_tokens,
            total_available,
            importance_weights
        )

        # Prepare results
        selected_chunks = [context_chunks[i] for i in selected_indices]
        if preserve_order:
            # Sort by original order
            ordered_indices = sorted(selected_indices)
            selected_chunks = [context_chunks[i] for i in ordered_indices]

        selected_tokens = sum(chunk_tokens[i] for i in selected_indices)

        metadata = {
            "total_tokens": query_tokens + selected_tokens,
            "selected_count": len(selected_indices),
            "total_count": len(context_chunks),
            "selection_ratio": len(selected_indices) / len(context_chunks),
            "token_utilization": selected_tokens / total_available,
            "selected_indices": list(selected_indices)
        }

        return {
            "selected_chunks": selected_chunks,
            "metadata": metadata
        }

    def _get_embeddings_cached(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with caching support."""
        embeddings = []
        to_compute = []
        to_compute_indices = []

        for idx, text in enumerate(texts):
            cached = self.embedding_cache.get(text)
            if cached is not None:
                embeddings.append(cached)
                continue

            embeddings.append(None)
            to_compute.append(text)
            to_compute_indices.append(idx)

        # Compute missing embeddings and update cache
        if to_compute:
            new_embeddings = self.embedding_model.encode(to_compute, batch_size=self.batch_size)
            for idx, text, emb in zip(to_compute_indices, to_compute, new_embeddings):
                embeddings[idx] = emb
                self.embedding_cache.set(text, emb)

        return np.array(embeddings)

    def _graft_selection(
        self,
        context_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        chunk_tokens: List[int],
        token_budget: int,
        importance_weights: Optional[List[float]] = None
    ) -> List[int]:
        """
        Adapted GRAFT selection algorithm for context trimming.
        """
        if len(context_embeddings) == 0:
            return []

        # Create combined embedding matrix [context; query]
        combined_embeddings = np.vstack([context_embeddings, query_embedding])

        # Perform SVD decomposition (adapted from feature_sel)
        U, S, Vt = np.linalg.svd(combined_embeddings.T, full_matrices=False)

        # Calculate target selection count based on token budget
        cumulative_tokens = np.cumsum(sorted(chunk_tokens))
        max_chunks = np.searchsorted(cumulative_tokens, token_budget, side='right')
        target_chunks = min(max_chunks, int(len(context_embeddings) * self.selection_fraction))

        if target_chunks == 0:
            return []

        # Adapted index selection using MaxVol technique
        selected_indices = self._maxvol_selection(
            Vt,
            target_chunks,
            context_embeddings,
            query_embedding,
            chunk_tokens,
            token_budget,
            importance_weights
        )

        return selected_indices

    def _maxvol_selection(
        self,
        Vt: np.ndarray,
        target_chunks: int,
        context_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        chunk_tokens: List[int],
        token_budget: int,
        importance_weights: Optional[List[float]] = None
    ) -> List[int]:
        """
        MaxVol-based selection adapted for context chunks.
        """
        n_chunks = len(context_embeddings)

        # Generate candidate ranks for selection
        min_rank = max(1, target_chunks - 5)
        max_rank = min(target_chunks + 5, n_chunks)
        candidate_ranks = list(range(min_rank, max_rank + 1))

        best_indices = []
        best_score = float('inf')

        for rank in candidate_ranks:
            # Use index_sel from decompositions module
            try:
                indices = index_sel(Vt, min(rank, Vt.shape[1]))
                indices = list(set(itertools.chain(*indices)))
                indices = [i for i in indices if i < n_chunks]  # Only context indices

                if not indices:
                    continue

                # Check token budget constraint
                selected_tokens = sum(chunk_tokens[i] for i in indices)
                if selected_tokens > token_budget:
                    # Greedily remove chunks until budget is met
                    indices = self._fit_to_budget(indices, chunk_tokens, token_budget)

                if not indices:
                    continue

                # Calculate importance score (lower is better)
                score = self._calculate_importance_score(
                    indices, context_embeddings, query_embedding, importance_weights
                )

                if score < best_score:
                    best_score = score
                    best_indices = indices

            except Exception as e:
                logging.debug(f"Error in rank {rank}: {e}")
                continue

        return best_indices

    def _fit_to_budget(self, indices: List[int], chunk_tokens: List[int], budget: int) -> List[int]:
        """Greedily remove chunks to fit within token budget."""
        if not indices:
            return []

        # Sort by token count (remove largest first)
        sorted_indices = sorted(indices, key=lambda i: chunk_tokens[i], reverse=True)

        selected = []
        total_tokens = 0

        for idx in sorted_indices:
            if total_tokens + chunk_tokens[idx] <= budget:
                selected.append(idx)
                total_tokens += chunk_tokens[idx]

        return selected

    def _calculate_importance_score(
        self,
        indices: List[int],
        context_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        importance_weights: Optional[List[float]] = None
    ) -> float:
        """
        Calculate importance score for selected chunks.
        Lower scores indicate better selections.
        """
        if not indices:
            return float('inf')

        selected_embeddings = context_embeddings[indices]

        # Calculate semantic similarity to query
        query_sim = np.mean([
            np.dot(emb, query_embedding.flatten()) /
            (np.linalg.norm(emb) * np.linalg.norm(query_embedding.flatten()))
            for emb in selected_embeddings
        ])

        # Calculate diversity within selection (higher diversity = lower penalty)
        diversity_penalty = 0
        if len(selected_embeddings) > 1:
            similarities = []
            for i in range(len(selected_embeddings)):
                for j in range(i+1, len(selected_embeddings)):
                    sim = np.dot(selected_embeddings[i], selected_embeddings[j]) / \
                          (np.linalg.norm(selected_embeddings[i]) * np.linalg.norm(selected_embeddings[j]))
                    similarities.append(abs(sim))
            diversity_penalty = np.mean(similarities) if similarities else 0

        # Apply importance weights if provided
        weight_bonus = 0
        if importance_weights:
            weight_bonus = np.mean([importance_weights[i] for i in indices])

        # Combine scores (lower is better)
        final_score = -query_sim + diversity_penalty - weight_bonus

        return final_score

    def batch_trim(
        self,
        contexts_and_queries: List[Tuple[List[str], str]],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Batch process multiple context-query pairs.

        Args:
            contexts_and_queries: List of (context_chunks, query) tuples
            show_progress: Whether to show progress bar

        Returns:
            List of trimming results
        """
        results = []
        iterator = tqdm(contexts_and_queries, desc="Trimming contexts") if show_progress else contexts_and_queries

        for context_chunks, query in iterator:
            result = self.trim_context(context_chunks, query)
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Get trimmer statistics."""
        return {
            "max_tokens": self.max_tokens,
            "selection_fraction": self.selection_fraction,
            "cache_stats": self.embedding_cache.get_stats(),
            "device": self.device
        }
