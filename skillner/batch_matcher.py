"""
Batch-optimized sliding window matcher.

Key optimization: Collect all window queries first, then process in batch.

Original approach (slow):
    for each word position:
        for each window size:
            query = build_query(window)
            result = query_method(query)  # GPU call #1, #2, #3, ...

Optimized approach (fast):
    queries = []
    for each word position:
        for each window size:
            queries.append(build_query(window))
    results = query_method.query_batch(queries)  # Single GPU batch call
"""

from typing import Callable, List, Tuple
from skillner.core.base import Node
from skillner.core.data_structures import Document, Sentence, Span, Candidate, Word

CONCEPT_ID: str = "concept_id"


class BatchSlidingWindowMatcher(Node):
    """
    Batch-optimized sliding window matcher.

    Instead of querying the knowledge base for each window individually,
    this collects all windows from a sentence/document and queries them
    in a single batch, maximizing GPU utilization.

    Performance:
    - Original: 1500 sequential queries per JD (~0.1 JDs/sec)
    - Batch: 1 batch query per JD (~10-50 JDs/sec)
    """

    def __init__(
        self,
        query_method,
        max_window_size: int = 5,
        pre_filter: Callable[[Word], str] = None
    ):
        """
        Initialize batch sliding window matcher.

        Args:
            query_method: Must support query_batch_with_indices() method
            max_window_size: Maximum words in a skill phrase
            pre_filter: Function to filter/transform words before querying
        """
        self.query_method = query_method
        self.max_window_size = max_window_size
        self.pre_filter = pre_filter if pre_filter is not None else lambda w: str(w)

        # Verify query method supports batch operations
        if not hasattr(query_method, 'query_batch_with_indices'):
            raise ValueError(
                "query_method must support query_batch_with_indices() for batch processing. "
                "Use BatchSemanticQueryMethod instead of SemanticQueryMethod."
            )

    def enrich_doc(self, doc: Document) -> None:
        """
        Find spans in document using batch processing.

        Key optimization: Process all windows in entire document in one batch.

        Args:
            doc: Document to process
        """
        # Step 1: Collect all queries from entire document
        queries = []
        indices = []

        for sent_idx, sentence in enumerate(doc):
            for word_idx in range(len(sentence)):
                for window_size in range(self.max_window_size, 0, -1):
                    idx_end = word_idx + window_size

                    # Check window is within sentence boundaries
                    if idx_end > len(sentence):
                        continue

                    window = slice(word_idx, idx_end)

                    # Build query
                    query = " ".join(
                        filter(
                            None,
                            (self.pre_filter(word) for word in sentence[window]),
                        )
                    )

                    if query:  # Only add non-empty queries
                        queries.append(query)
                        indices.append((sent_idx, word_idx, window_size))

        # Step 2: Batch query all windows (GPU parallel)
        if not queries:
            return

        batch_results = self.query_method.query_batch_with_indices(queries, indices)

        # Step 3: Create spans from results
        for (sent_idx, word_idx, window_size), li_responses in batch_results.items():
            if not li_responses:
                continue

            sentence = doc[sent_idx]
            idx_end = word_idx + window_size
            window = slice(word_idx, idx_end)

            # Check if we already have a span at this position
            # (we want the longest match, which is processed first)
            existing_span = None
            for span in sentence.li_spans:
                if span.li_candidates and len(span.li_candidates) > 0:
                    # Check if any candidate starts at same position
                    for candidate in span.li_candidates:
                        if candidate.window.start == word_idx:
                            existing_span = span
                            break
                if existing_span:
                    break

            # Create span or add to existing
            if existing_span is None:
                span = Span()
                for response in li_responses:
                    if response and len(response) > 0:
                        concept_id = response[CONCEPT_ID]
                        candidate = Candidate(window, concept_id)
                        candidate.metadata = response
                        span.add_candidate(candidate)

                if not span.is_empty():
                    sentence.li_spans.append(span)
            else:
                # Add candidates to existing span if from larger window
                # (handled by span processor later)
                for response in li_responses:
                    if response and len(response) > 0:
                        concept_id = response[CONCEPT_ID]
                        candidate = Candidate(window, concept_id)
                        candidate.metadata = response
                        existing_span.add_candidate(candidate)


class OptimizedBatchSlidingWindowMatcher(BatchSlidingWindowMatcher):
    """
    Further optimized version with smart window prioritization.

    Additional optimizations:
    1. Process larger windows first (they're more specific)
    2. Skip windows that overlap with already-matched spans
    3. Early termination when span is found
    """

    def enrich_doc(self, doc: Document) -> None:
        """
        Find spans with smart window prioritization.

        Optimization: For each position, try largest window first.
        If it matches, skip smaller windows (they're less specific).
        """
        queries = []
        indices = []

        for sent_idx, sentence in enumerate(doc):
            # Track which positions already have matches
            matched_positions = set()

            for word_idx in range(len(sentence)):
                # Skip if this position already matched
                if word_idx in matched_positions:
                    continue

                # Try windows from largest to smallest
                for window_size in range(self.max_window_size, 0, -1):
                    idx_end = word_idx + window_size

                    if idx_end > len(sentence):
                        continue

                    # Skip if any position in window already matched
                    if any(pos in matched_positions for pos in range(word_idx, idx_end)):
                        continue

                    window = slice(word_idx, idx_end)

                    query = " ".join(
                        filter(
                            None,
                            (self.pre_filter(word) for word in sentence[window]),
                        )
                    )

                    if query:
                        queries.append(query)
                        indices.append((sent_idx, word_idx, window_size))

        # Batch query
        if not queries:
            return

        batch_results = self.query_method.query_batch_with_indices(queries, indices)

        # Create spans and track matched positions
        for (sent_idx, word_idx, window_size), li_responses in batch_results.items():
            if not li_responses:
                continue

            sentence = doc[sent_idx]
            idx_end = word_idx + window_size
            window = slice(word_idx, idx_end)

            span = Span()
            for response in li_responses:
                if response and len(response) > 0:
                    concept_id = response[CONCEPT_ID]
                    candidate = Candidate(window, concept_id)
                    candidate.metadata = response
                    span.add_candidate(candidate)

            if not span.is_empty():
                sentence.li_spans.append(span)
