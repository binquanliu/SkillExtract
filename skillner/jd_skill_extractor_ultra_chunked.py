"""
Ultra-optimized extractor with chunked processing for large datasets.

This version:
- Processes JDs in chunks (e.g., 500 at a time)
- Applies ultra-optimization within each chunk
- Prevents memory explosion on large datasets
- Maintains high throughput

Performance: 15-30 JDs/sec (chunk-based multi-JD batching)
Stability: Can process unlimited JDs
"""

from skillner.jd_skill_extractor_ultra import UltraOptimizedSkillExtractor
from typing import List, Dict
from tqdm import tqdm
import gc
import torch


class UltraOptimizedSkillExtractorChunked(UltraOptimizedSkillExtractor):
    """
    Chunked version of UltraOptimizedSkillExtractor.

    Instead of processing all JDs at once (which causes OOM),
    this processes them in chunks of configurable size.

    Example:
        100K JDs → 200 chunks of 500 JDs each
        Each chunk uses ultra-optimization
        Between chunks: memory cleanup
    """

    def __init__(
        self,
        kb_path: str,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.6,
        max_window_size: int = 5,
        batch_size: int = 4096,
        use_fp16: bool = True,
        cache_queries: bool = True,
        chunk_size: int = 500  # Process 500 JDs at a time
    ):
        """
        Initialize chunked ultra extractor.

        Args:
            chunk_size: Number of JDs to process in each chunk
                - Smaller = more stable, slower
                - Larger = faster, more memory
                - Recommended: 200-1000 depending on JD size
        """
        super().__init__(
            kb_path=kb_path,
            model_name=model_name,
            similarity_threshold=similarity_threshold,
            max_window_size=max_window_size,
            batch_size=batch_size,
            use_fp16=use_fp16,
            cache_queries=cache_queries
        )

        self.chunk_size = chunk_size
        print(f"✓ Chunked processing enabled (chunk_size={chunk_size})\n")

    def extract_skills_batch_chunked(
        self,
        job_descriptions: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process JDs in chunks with ultra-optimization per chunk.

        Args:
            job_descriptions: List of all JD texts
            show_progress: Show progress bar

        Returns:
            List of extraction results
        """
        total_jds = len(job_descriptions)
        num_chunks = (total_jds + self.chunk_size - 1) // self.chunk_size

        if show_progress:
            print(f"Processing {total_jds:,} JDs in {num_chunks} chunks of ~{self.chunk_size}")

        all_results = []

        # Process chunks
        chunk_iterator = range(0, total_jds, self.chunk_size)
        if show_progress:
            chunk_iterator = tqdm(
                chunk_iterator,
                desc="Processing chunks",
                total=num_chunks,
                unit="chunk"
            )

        for chunk_start in chunk_iterator:
            chunk_end = min(chunk_start + self.chunk_size, total_jds)
            chunk_jds = job_descriptions[chunk_start:chunk_end]

            if show_progress:
                tqdm.write(f"\nChunk {chunk_start//self.chunk_size + 1}/{num_chunks}: "
                          f"JDs {chunk_start:,} - {chunk_end:,}")

            # Process chunk with ultra-optimization
            chunk_results = self.extract_skills_batch_ultra(
                chunk_jds,
                show_progress=False  # Don't show per-chunk progress
            )

            all_results.extend(chunk_results)

            # Cleanup after each chunk
            self._cleanup_chunk()

            if show_progress:
                processed = len(all_results)
                progress_pct = (processed / total_jds) * 100
                tqdm.write(f"  → Processed {processed:,}/{total_jds:,} ({progress_pct:.1f}%)")

        # Final cleanup
        self._cleanup_chunk()

        if show_progress:
            print(f"\n✓ Completed: {len(all_results):,} JDs processed")

        return all_results

    def extract_skills(
        self,
        job_description,
        show_progress: bool = False,
        return_details: bool = True
    ):
        """
        Extract skills with automatic chunking for large batches.

        Args:
            job_description: Single JD or list of JDs
            show_progress: Show progress bar
            return_details: Return detailed results

        Returns:
            Dict (single) or List[Dict] (batch)
        """
        if isinstance(job_description, list):
            # Use chunked processing for batches
            return self.extract_skills_batch_chunked(
                job_description,
                show_progress=show_progress
            )
        else:
            # Single JD - no chunking needed
            return super().extract_skills(
                job_description,
                show_progress=False,
                return_details=return_details
            )

    def extract_skills_batch(
        self,
        job_descriptions: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """Alias for chunked batch processing."""
        return self.extract_skills_batch_chunked(job_descriptions, show_progress)

    def _cleanup_chunk(self):
        """Cleanup after processing a chunk."""
        # Clear Python garbage
        gc.collect()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Optionally clear query cache to save memory
        # (Comment out if you want to keep cache across chunks)
        # if hasattr(self, 'query_cache'):
        #     self.query_cache.clear()


if __name__ == '__main__':
    print("Ultra-Optimized Chunked Skill Extractor Demo\n")

    extractor = UltraOptimizedSkillExtractorChunked(
        kb_path='.skillner-kb/MERGED_EN.pkl',
        batch_size=4096,
        use_fp16=True,
        cache_queries=True,
        chunk_size=500
    )

    # Simulate large dataset
    test_jds = [
        "Data Scientist with Python and machine learning skills.",
        "Software Engineer proficient in Java and AWS.",
        "Marketing Manager with digital marketing expertise."
    ] * 1000  # 3000 JDs

    print("Processing 3000 job descriptions in chunks...")
    import time
    start = time.time()
    results = extractor.extract_skills_batch_chunked(test_jds, show_progress=True)
    elapsed = time.time() - start

    print(f"\n✓ Processed {len(results)} JDs in {elapsed:.2f}s")
    print(f"  Throughput: {len(results)/elapsed:.1f} JDs/sec")
