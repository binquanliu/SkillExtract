"""
Batch processing for large-scale job description skill extraction.

This script handles millions of job descriptions efficiently:
- Chunk processing with progress tracking
- Multi-processing support
- Resume capability (saves checkpoints)
- Memory-efficient parquet I/O

Usage:
    python batch_process_jd.py --input data.parquet --output results.parquet --sample-size 50
"""

import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse
import time
from typing import List, Dict

from skillner.onet_converter import load_kb
from skillner.enhanced_matching import FuzzyQueryMethod  # 比语义匹配快很多
from skillner.core import Pipeline, Document
from skillner.text_loaders import StrTextLoader
from skillner.matchers import SlidingWindowMatcher
from skillner.conflict_resolvers import SpanProcessor


def extract_skills_from_jd(job_description: str, query_method) -> List[Dict]:
    """Extract skills from a single job description."""
    doc = Document()
    pipeline = Pipeline()

    pipeline.add_node(StrTextLoader(job_description), name='loader')
    pipeline.add_node(
        SlidingWindowMatcher(
            query_method,
            max_window_size=5,
            pre_filter=lambda w: w.lower()
        ),
        name='matcher'
    )
    pipeline.add_node(
        SpanProcessor(
            dict_filters={'max_candidate': lambda span: max(span.li_candidates, key=len)}
        ),
        name='resolver'
    )

    pipeline.run(doc)

    skills = []
    for sentence in doc:
        for span in sentence.li_spans:
            candidate = span.metadata.get('max_candidate')
            if candidate:
                skills.append({
                    'skill': candidate.metadata['pref_label'],
                    'section': candidate.metadata.get('section'),
                    'concept_id': candidate.metadata.get('concept_id')
                })

    return skills


def process_batch(batch_df: pd.DataFrame, kb_path: str, use_fuzzy: bool = True) -> pd.DataFrame:
    """
    Process a batch of job descriptions.

    Args:
        batch_df: DataFrame with job descriptions
        kb_path: Path to knowledge base
        use_fuzzy: Use fuzzy matching (faster) vs semantic (slower but better)

    Returns:
        DataFrame with extracted skills
    """
    # Load KB (each process loads its own)
    kb = load_kb(kb_path)

    # Create query method
    if use_fuzzy:
        query_method = FuzzyQueryMethod(kb, similarity_threshold=0.85)
    else:
        from skillner.enhanced_matching import SemanticQueryMethod
        query_method = SemanticQueryMethod(kb, similarity_threshold=0.6)

    results = []

    for idx, row in batch_df.iterrows():
        try:
            skills = extract_skills_from_jd(row['job_description'], query_method)

            # Aggregate by section
            skills_by_section = {}
            for skill in skills:
                section = skill.get('section', 'Unknown')
                if section not in skills_by_section:
                    skills_by_section[section] = []
                skills_by_section[section].append(skill['skill'])

            results.append({
                'index': idx,
                'onet_code': row['onet_code'],
                'post_date': row['post_date'],
                'quarter': row.get('quarter'),
                'num_skills_total': len(skills),
                'num_skills_unique': len(set(s['skill'] for s in skills)),
                'skills_by_section': skills_by_section,
                'all_skills': [s['skill'] for s in skills]
            })

        except Exception as e:
            results.append({
                'index': idx,
                'onet_code': row['onet_code'],
                'post_date': row['post_date'],
                'error': str(e)
            })

    return pd.DataFrame(results)


def batch_process_with_checkpoints(
    input_path: str,
    output_path: str,
    kb_path: str,
    batch_size: int = 1000,
    use_fuzzy: bool = True,
    resume: bool = True
):
    """
    Process large dataset with checkpoints.

    Args:
        input_path: Input parquet file
        output_path: Output parquet file
        kb_path: Knowledge base path
        batch_size: Number of records per batch
        use_fuzzy: Use fuzzy (fast) vs semantic (slow)
        resume: Resume from checkpoint if exists
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)

    print(f"Total records: {len(df):,}")

    # Create quarter column if not exists
    if 'quarter' not in df.columns:
        df['post_date'] = pd.to_datetime(df['post_date'])
        df['quarter'] = df['post_date'].dt.to_period('Q')

    # Checkpoint file
    checkpoint_path = output_path.replace('.parquet', '_checkpoint.parquet')

    # Resume from checkpoint
    if resume and Path(checkpoint_path).exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        processed_df = pd.read_parquet(checkpoint_path)
        processed_indices = set(processed_df['index'].values)
        df = df[~df.index.isin(processed_indices)]
        print(f"Remaining records: {len(df):,}")
    else:
        processed_df = None

    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size

    print(f"\nProcessing in {total_batches} batches of {batch_size}...")
    print(f"Method: {'Fuzzy matching' if use_fuzzy else 'Semantic matching'}")

    all_results = []

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i+batch_size]

        # Process batch
        start_time = time.time()
        batch_results = process_batch(batch, kb_path, use_fuzzy)
        elapsed = time.time() - start_time

        all_results.append(batch_results)

        # Save checkpoint every 10 batches
        if (i // batch_size + 1) % 10 == 0:
            checkpoint_df = pd.concat(all_results, ignore_index=True)
            if processed_df is not None:
                checkpoint_df = pd.concat([processed_df, checkpoint_df], ignore_index=True)

            checkpoint_df.to_parquet(checkpoint_path, index=False)
            print(f"\nCheckpoint saved: {len(checkpoint_df):,} records")

        # Print progress
        avg_time = elapsed / len(batch)
        remaining_batches = total_batches - (i // batch_size + 1)
        est_time = remaining_batches * batch_size * avg_time / 3600

        tqdm.write(f"Batch {i//batch_size + 1}/{total_batches} | "
                  f"{avg_time:.2f}s/record | "
                  f"Est. remaining: {est_time:.1f}h")

    # Final save
    print("\nCombining results...")
    final_df = pd.concat(all_results, ignore_index=True)
    if processed_df is not None:
        final_df = pd.concat([processed_df, final_df], ignore_index=True)

    final_df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved results to {output_path}")
    print(f"  Total processed: {len(final_df):,} records")

    # Clean up checkpoint
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return final_df


def main():
    parser = argparse.ArgumentParser(description='Batch process job descriptions')
    parser.add_argument('--input', required=True, help='Input parquet file')
    parser.add_argument('--output', required=True, help='Output parquet file')
    parser.add_argument('--kb', default='.skillner-kb/ONET_EN.pkl', help='Knowledge base path')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size')
    parser.add_argument('--method', choices=['fuzzy', 'semantic'], default='fuzzy',
                       help='Matching method')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size per onet_code-quarter group')
    parser.add_argument('--no-resume', action='store_true', help='Do not resume from checkpoint')

    args = parser.parse_args()

    # Load and sample if requested
    df = pd.read_parquet(args.input)

    if args.sample_size:
        print(f"Sampling {args.sample_size} records per onet_code-quarter...")
        df['post_date'] = pd.to_datetime(df['post_date'])
        df['quarter'] = df['post_date'].dt.to_period('Q')

        df = df.groupby(['onet_code', 'quarter']).apply(
            lambda x: x.sample(min(args.sample_size, len(x)), random_state=42)
        ).reset_index(drop=True)

        print(f"Sampled dataset: {len(df):,} records")

        # Save sampled data
        sampled_path = args.input.replace('.parquet', '_sampled.parquet')
        df.to_parquet(sampled_path, index=False)
        args.input = sampled_path

    # Process
    batch_process_with_checkpoints(
        input_path=args.input,
        output_path=args.output,
        kb_path=args.kb,
        batch_size=args.batch_size,
        use_fuzzy=(args.method == 'fuzzy'),
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
