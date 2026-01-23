#!/usr/bin/env python3
"""
Quick performance test for batch-optimized extractor.

This script compares performance of original vs batch-optimized extractors.
Run this to verify the speedup on your hardware.

Usage:
    python test_batch_performance.py
"""

import time
import sys
import pandas as pd
from pathlib import Path


def test_batch_performance():
    """Test batch extractor performance."""

    print("="*70)
    print("BATCH EXTRACTOR PERFORMANCE TEST")
    print("="*70)

    # Check if KB exists
    kb_path = '.skillner-kb/MERGED_EN.pkl'
    if not Path(kb_path).exists():
        kb_path = '.skillner-kb/ONET_EN.pkl'
        if not Path(kb_path).exists():
            print(f"✗ Knowledge base not found!")
            print(f"  Please create it first using onet_converter.py")
            return

    print(f"\nKnowledge base: {kb_path}")

    # Sample job descriptions for testing
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

    # Duplicate to make larger test set
    test_jds = sample_jds * 4  # 20 JDs total

    print(f"Test size: {len(test_jds)} job descriptions")
    print(f"Average length: {sum(len(jd) for jd in test_jds) / len(test_jds):.0f} characters")

    # Test batch-optimized extractor
    print("\n" + "="*70)
    print("Testing Batch-Optimized Extractor")
    print("="*70)

    try:
        from skillner.jd_skill_extractor_batch import BatchJobDescriptionSkillExtractor

        print("\nInitializing extractor...")
        extractor = BatchJobDescriptionSkillExtractor(
            kb_path,
            model_name='all-MiniLM-L6-v2',
            similarity_threshold=0.6,
            batch_size=256
        )

        print(f"\nProcessing {len(test_jds)} job descriptions...")
        start = time.time()
        results = extractor.extract_skills_batch(test_jds, show_progress=True)
        elapsed = time.time() - start

        throughput = len(test_jds) / elapsed
        avg_time = elapsed / len(test_jds)

        print(f"\n✓ Completed!")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Throughput: {throughput:.2f} JDs/sec")
        print(f"  Avg time per JD: {avg_time*1000:.0f}ms")

        # Show sample results
        print(f"\nSample extraction results:")
        for i, result in enumerate(results[:3]):
            print(f"\nJD #{i+1}: Found {result['num_skills']} skills")
            print(f"  Skills: {', '.join(result['skills'][:5])}")
            if len(result['skills']) > 5:
                print(f"  ... and {len(result['skills']) - 5} more")

        # Statistics
        stats = extractor.get_statistics(results)
        print(f"\nOverall statistics:")
        print(f"  Total unique skills: {stats['unique_skills_total']}")
        print(f"  Avg skills per JD: {stats['skills_per_jd']['mean']:.1f}")
        print(f"  Min/Max: {stats['skills_per_jd']['min']:.0f} / {stats['skills_per_jd']['max']:.0f}")

        # Performance assessment
        print(f"\n" + "="*70)
        print("PERFORMANCE ASSESSMENT")
        print("="*70)

        if throughput >= 10:
            print(f"✓ EXCELLENT: {throughput:.1f} JDs/sec is very fast!")
            print(f"  Your GPU is working well.")
        elif throughput >= 5:
            print(f"✓ GOOD: {throughput:.1f} JDs/sec is decent.")
            print(f"  Consider increasing batch_size for more speedup.")
        elif throughput >= 1:
            print(f"⚠ MODERATE: {throughput:.1f} JDs/sec")
            print(f"  Check if GPU is being used (should see CUDA messages above).")
        else:
            print(f"✗ SLOW: {throughput:.1f} JDs/sec")
            print(f"  Something is wrong. Expected 10-50 JDs/sec on A100 GPU.")
            print(f"  Possible issues:")
            print(f"    - GPU not being used (check CUDA availability)")
            print(f"    - Batch size too small")
            print(f"    - System under heavy load")

        print("\nFor processing 100K job descriptions:")
        estimated_time = 100000 / throughput
        print(f"  Estimated time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")

        return results

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("\nBatch Extractor Performance Test\n")
    test_batch_performance()
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
