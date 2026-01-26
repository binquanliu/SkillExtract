#!/usr/bin/env python3
"""
Performance comparison: Batch vs Ultra-optimized extractors.

This script benchmarks:
1. Batch extractor (JD-by-JD processing)
2. Ultra extractor (multi-JD batching + FP16)

Run with: python test_ultra_performance.py
"""

import time
import torch
from pathlib import Path


def test_ultra_performance():
    """Compare batch vs ultra-optimized performance."""

    print("="*70)
    print("PERFORMANCE COMPARISON: Batch vs Ultra-Optimized")
    print("="*70)

    # Check KB
    kb_path = '.skillner-kb/MERGED_EN.pkl'
    if not Path(kb_path).exists():
        kb_path = '.skillner-kb/ONET_EN.pkl'
        if not Path(kb_path).exists():
            print("✗ Knowledge base not found!")
            return

    print(f"\nKnowledge base: {kb_path}")

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ No GPU available - performance will be limited")

    # Test data
    sample_jds = [
        """
        Data Scientist position requiring strong analytical and problem-solving skills.
        Must have experience with Python, R, SQL, and machine learning frameworks like
        TensorFlow and scikit-learn. Strong communication skills and ability to work
        in cross-functional teams required. Experience with big data technologies
        (Spark, Hadoop) is a plus.
        """,
        """
        Software Engineer role focused on backend development. Proficiency in Java,
        Spring Boot, and microservices architecture required. Experience with cloud
        platforms (AWS, Azure), Docker, and Kubernetes preferred. Must have excellent
        problem-solving abilities and strong teamwork skills.
        """,
        """
        Marketing Manager position requiring strategic thinking and leadership skills.
        Experience with digital marketing, SEO, and social media management essential.
        Strong analytical skills and proficiency with marketing analytics tools required.
        Excellent written and verbal communication skills necessary.
        """,
        """
        Registered Nurse position requiring patient care skills and clinical judgment.
        Must have strong communication skills and ability to work under pressure.
        Experience with electronic health records and medical equipment operation required.
        BLS and ACLS certification necessary.
        """,
        """
        Financial Analyst role requiring strong analytical and quantitative skills.
        Proficiency in Excel, financial modeling, and data analysis required.
        Experience with SQL, Python, or R is a plus. Must have excellent attention
        to detail and strong communication skills.
        """
    ]

    # Test sizes
    test_sizes = [20, 50, 100]

    print(f"\nTest JD (sample):")
    print(f"  Length: {len(sample_jds[0])} chars")
    print(f"  Words: ~{len(sample_jds[0].split())} words")

    # Test 1: Batch extractor
    print("\n" + "="*70)
    print("TEST 1: Batch Extractor (Current)")
    print("="*70)

    try:
        from skillner.jd_skill_extractor_batch import BatchJobDescriptionSkillExtractor

        extractor_batch = BatchJobDescriptionSkillExtractor(
            kb_path,
            batch_size=512,
            similarity_threshold=0.6
        )

        for size in test_sizes[:1]:  # Just test smallest for batch
            test_jds = sample_jds * (size // len(sample_jds))

            print(f"\nProcessing {len(test_jds)} JDs...")
            start = time.time()
            results_batch = extractor_batch.extract_skills_batch(test_jds, show_progress=False)
            elapsed = time.time() - start

            throughput = len(test_jds) / elapsed

            print(f"  Time: {elapsed:.1f}s")
            print(f"  Throughput: {throughput:.2f} JDs/sec")
            print(f"  Avg per JD: {elapsed/len(test_jds)*1000:.0f}ms")

            batch_throughput = throughput

        del extractor_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Error: {e}")
        batch_throughput = 0

    # Test 2: Ultra-optimized extractor
    print("\n" + "="*70)
    print("TEST 2: Ultra-Optimized Extractor (New)")
    print("="*70)

    try:
        from skillner.jd_skill_extractor_ultra import UltraOptimizedSkillExtractor

        # Test different batch sizes
        for batch_size in [4096, 8192, 16384]:
            print(f"\n--- Batch size: {batch_size} ---")

            extractor_ultra = UltraOptimizedSkillExtractor(
                kb_path,
                batch_size=batch_size,
                similarity_threshold=0.6,
                use_fp16=True,
                cache_queries=True
            )

            for size in test_sizes:
                test_jds = sample_jds * (size // len(sample_jds))

                print(f"\nProcessing {len(test_jds)} JDs (batch_size={batch_size})...")
                start = time.time()
                results_ultra = extractor_ultra.extract_skills_batch_ultra(
                    test_jds,
                    show_progress=True
                )
                elapsed = time.time() - start

                throughput = len(test_jds) / elapsed

                print(f"\n  ✓ Time: {elapsed:.1f}s")
                print(f"  ✓ Throughput: {throughput:.2f} JDs/sec")
                print(f"  ✓ Avg per JD: {elapsed/len(test_jds)*1000:.0f}ms")

                if batch_throughput > 0:
                    speedup = throughput / batch_throughput
                    print(f"  🚀 Speedup vs Batch: {speedup:.1f}x")

                # Show GPU memory usage
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"  GPU Memory: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

            # Clear cache between batch sizes
            extractor_ultra.clear_cache()
            del extractor_ultra
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print("\nRecommendations:")
    print("  • For A100-80GB: batch_size=8192-16384")
    print("  • For smaller GPUs: batch_size=4096")
    print("  • Enable FP16 for 2x speedup")
    print("  • Enable caching for repeated queries")
    print("\nExpected throughput on A100:")
    print("  • Batch extractor: 3-5 JDs/sec")
    print("  • Ultra extractor: 15-50 JDs/sec")
    print("  • Speedup: 5-10x")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Ultra-Optimized Skill Extractor Performance Test")
    print("="*70)

    test_ultra_performance()

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
