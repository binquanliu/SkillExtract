"""
Analyze Extracted Skills (Multi-File Version)

Fixes:
1. Memory-efficient: reads files one by one, not all at once
2. Handles string-formatted lists in parquet (e.g., "['skill1', 'skill2']")
3. Handles empty data gracefully (no divide by zero, no IndexError)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json
import ast
import gc
from typing import List, Dict, Any, Optional

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available, skipping plots")


# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

INPUT_FOLDER = '../data/extracted_skills_production'  # Folder with parquet files
SKILLS_COLUMN = 'skills'
NUM_SKILLS_COLUMN = 'num_skills'
BY_SECTION_COLUMN = 'by_section'
ONET_CODE_COLUMN = 'onet_code'

# Processing settings
FILES_PER_BATCH = 5  # Process N files at a time to control memory
TOP_N_SKILLS = 30


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_skills_column(value) -> List[str]:
    """
    Parse skills column that might be stored as string or list.

    Handles:
    - Already a list: ['skill1', 'skill2']
    - String representation: "['skill1', 'skill2']"
    - None/NaN
    - Empty list: []
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value in ('[]', '', 'nan', 'None'):
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def parse_by_section_column(value) -> Dict[str, List[str]]:
    """Parse by_section column that might be stored as string or dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if value in ('{}', '', 'nan', 'None'):
            return {}
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return {}


def process_single_file(filepath: Path) -> Dict[str, Any]:
    """
    Process a single parquet file and extract statistics.

    Returns aggregated counts, not full data, to save memory.
    """
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"    Error reading {filepath.name}: {e}")
        return None

    stats = {
        'filepath': str(filepath),
        'num_rows': len(df),
        'skill_counter': Counter(),
        'section_counts': {},
        'section_jd_counts': {},
        'num_skills_list': [],
        'onet_counts': Counter(),
        'jds_with_skills': 0,
        'jds_without_skills': 0,
    }

    # Process skills column
    if SKILLS_COLUMN in df.columns:
        for idx, row in df.iterrows():
            skills = parse_skills_column(row.get(SKILLS_COLUMN))
            num_skills = len(skills)

            stats['num_skills_list'].append(num_skills)
            stats['skill_counter'].update(skills)

            if num_skills > 0:
                stats['jds_with_skills'] += 1
            else:
                stats['jds_without_skills'] += 1

    # Process by_section column
    if BY_SECTION_COLUMN in df.columns:
        for idx, row in df.iterrows():
            by_section = parse_by_section_column(row.get(BY_SECTION_COLUMN))
            for section, skills in by_section.items():
                if section not in stats['section_counts']:
                    stats['section_counts'][section] = 0
                    stats['section_jd_counts'][section] = 0
                stats['section_counts'][section] += len(skills)
                stats['section_jd_counts'][section] += 1

    # Process ONET column
    if ONET_CODE_COLUMN in df.columns:
        stats['onet_counts'].update(df[ONET_CODE_COLUMN].dropna().tolist())

    # Free memory
    del df
    gc.collect()

    return stats


def merge_stats(all_stats: List[Dict]) -> Dict[str, Any]:
    """Merge statistics from multiple files."""
    merged = {
        'total_rows': 0,
        'total_files': len(all_stats),
        'skill_counter': Counter(),
        'section_counts': {},
        'section_jd_counts': {},
        'all_num_skills': [],
        'onet_counts': Counter(),
        'jds_with_skills': 0,
        'jds_without_skills': 0,
    }

    for stats in all_stats:
        if stats is None:
            continue

        merged['total_rows'] += stats['num_rows']
        merged['skill_counter'].update(stats['skill_counter'])
        merged['onet_counts'].update(stats['onet_counts'])
        merged['all_num_skills'].extend(stats['num_skills_list'])
        merged['jds_with_skills'] += stats['jds_with_skills']
        merged['jds_without_skills'] += stats['jds_without_skills']

        for section, count in stats['section_counts'].items():
            if section not in merged['section_counts']:
                merged['section_counts'][section] = 0
                merged['section_jd_counts'][section] = 0
            merged['section_counts'][section] += count
            merged['section_jd_counts'][section] += stats['section_jd_counts'].get(section, 0)

    return merged


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print("MULTI-FILE SKILL ANALYSIS (Memory Efficient)")
    print("="*70)

    # Find all parquet files
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        print(f"\n[ERROR] Folder not found: {INPUT_FOLDER}")
        print("Please update INPUT_FOLDER in the configuration section.")
        return

    parquet_files = sorted(input_path.glob('*.parquet'))

    if not parquet_files:
        print(f"\n[ERROR] No parquet files found in {INPUT_FOLDER}")
        return

    print(f"\nFound {len(parquet_files)} parquet files")
    print(f"Processing {FILES_PER_BATCH} files at a time...\n")

    # Process files in batches
    all_stats = []

    for i, filepath in enumerate(parquet_files):
        print(f"  [{i+1}/{len(parquet_files)}] Processing {filepath.name}...", end=" ")
        stats = process_single_file(filepath)
        if stats:
            all_stats.append(stats)
            print(f"OK ({stats['num_rows']:,} rows, {stats['jds_with_skills']:,} with skills)")
        else:
            print("FAILED")

        # Periodic memory cleanup
        if (i + 1) % FILES_PER_BATCH == 0:
            gc.collect()

    if not all_stats:
        print("\n[ERROR] No files were processed successfully!")
        return

    # Merge all statistics
    print("\nMerging statistics...")
    merged = merge_stats(all_stats)

    # Free individual stats
    del all_stats
    gc.collect()

    # =================================================================
    # REPORT
    # =================================================================

    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)

    print(f"\nTotal files processed: {merged['total_files']}")
    print(f"Total job descriptions: {merged['total_rows']:,}")
    print(f"JDs with skills: {merged['jds_with_skills']:,} ({merged['jds_with_skills']/merged['total_rows']*100:.1f}%)")
    print(f"JDs without skills: {merged['jds_without_skills']:,} ({merged['jds_without_skills']/merged['total_rows']*100:.1f}%)")

    # Skills per JD statistics
    num_skills_arr = np.array(merged['all_num_skills'])
    if len(num_skills_arr) > 0:
        print(f"\nSkills per Job Description:")
        print(f"  Mean:   {np.mean(num_skills_arr):.1f}")
        print(f"  Median: {np.median(num_skills_arr):.1f}")
        print(f"  Std:    {np.std(num_skills_arr):.1f}")
        print(f"  Min:    {np.min(num_skills_arr):.0f}")
        print(f"  Max:    {np.max(num_skills_arr):.0f}")

    # Unique skills
    skill_counter = merged['skill_counter']
    if skill_counter:
        unique_skills = len(skill_counter)
        total_mentions = sum(skill_counter.values())
        print(f"\nSkill diversity:")
        print(f"  Total unique skills: {unique_skills:,}")
        print(f"  Total skill mentions: {total_mentions:,}")
        if unique_skills > 0:
            print(f"  Avg mentions per skill: {total_mentions/unique_skills:.1f}")

    # =================================================================
    # TOP SKILLS
    # =================================================================

    print("\n" + "="*70)
    print(f"TOP {TOP_N_SKILLS} MOST COMMON SKILLS")
    print("="*70)

    if skill_counter:
        top_skills = skill_counter.most_common(TOP_N_SKILLS)
        print(f"\n{'Rank':<6}{'Skill':<50}{'Count':>10}{'% of JDs':>12}")
        print("-"*78)

        for rank, (skill, count) in enumerate(top_skills, 1):
            pct = (count / merged['total_rows']) * 100
            print(f"{rank:<6}{skill:<50}{count:>10,}{pct:>11.1f}%")
    else:
        print("\n[WARNING] No skills found in the data!")
        print("Possible causes:")
        print("  1. Skills column contains strings instead of lists")
        print("  2. Skills column is empty")
        print("  3. Column name mismatch (check SKILLS_COLUMN setting)")

    # =================================================================
    # SKILLS BY SECTION
    # =================================================================

    print("\n" + "="*70)
    print("SKILLS BY SECTION")
    print("="*70)

    section_counts = merged['section_counts']
    section_jd_counts = merged['section_jd_counts']

    if section_counts:
        print(f"\n{'Section':<30}{'Total Skills':>15}{'% of Total':>12}")
        print("-"*57)

        total_section_skills = sum(section_counts.values())
        for section in sorted(section_counts.keys(), key=lambda x: section_counts[x], reverse=True):
            count = section_counts[section]
            pct = (count / total_section_skills * 100) if total_section_skills > 0 else 0
            print(f"{section:<30}{count:>15,}{pct:>11.1f}%")
    else:
        print("\n[INFO] No section data available")

    # =================================================================
    # TOP ONET CODES
    # =================================================================

    print("\n" + "="*70)
    print("TOP ONET CODES")
    print("="*70)

    onet_counts = merged['onet_counts']
    if onet_counts:
        top_onet = onet_counts.most_common(10)
        print(f"\n{'ONET Code':<20}{'Count':>10}{'% of Total':>12}")
        print("-"*42)

        for onet_code, count in top_onet:
            pct = (count / merged['total_rows']) * 100
            print(f"{onet_code:<20}{count:>10,}{pct:>11.1f}%")
    else:
        print("\n[INFO] No ONET code data available")

    # =================================================================
    # SAVE SUMMARY
    # =================================================================

    summary_path = Path(INPUT_FOLDER) / 'analysis_summary.json'
    summary = {
        'total_files': merged['total_files'],
        'total_jds': merged['total_rows'],
        'jds_with_skills': merged['jds_with_skills'],
        'jds_without_skills': merged['jds_without_skills'],
        'unique_skills': len(skill_counter) if skill_counter else 0,
        'top_30_skills': [(s, c) for s, c in skill_counter.most_common(30)] if skill_counter else [],
        'skills_by_section': dict(section_counts),
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_path}")

    # =================================================================
    # OPTIONAL PLOTS
    # =================================================================

    if HAS_PLOTTING and skill_counter:
        print("\nGenerating plots...")

        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Plot 1: Top skills bar chart
            top_20 = skill_counter.most_common(20)
            if top_20:
                skills_names = [s[0][:40] for s in top_20]  # Truncate long names
                skills_counts = [s[1] for s in top_20]

                axes[0, 0].barh(range(len(skills_names)), skills_counts, color='steelblue')
                axes[0, 0].set_yticks(range(len(skills_names)))
                axes[0, 0].set_yticklabels(skills_names, fontsize=8)
                axes[0, 0].set_xlabel('Count')
                axes[0, 0].set_title('Top 20 Most Common Skills')
                axes[0, 0].invert_yaxis()

            # Plot 2: Skills per JD histogram
            if len(num_skills_arr) > 0:
                axes[0, 1].hist(num_skills_arr, bins=50, edgecolor='black', alpha=0.7)
                axes[0, 1].axvline(np.mean(num_skills_arr), color='red', linestyle='--',
                                   label=f'Mean: {np.mean(num_skills_arr):.1f}')
                axes[0, 1].set_xlabel('Number of Skills')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of Skills per JD')
                axes[0, 1].legend()

            # Plot 3: Skills by section
            if section_counts:
                sections = list(section_counts.keys())
                counts = [section_counts[s] for s in sections]

                axes[1, 0].bar(sections, counts, color='coral', edgecolor='black', alpha=0.7)
                axes[1, 0].set_xlabel('Section')
                axes[1, 0].set_ylabel('Total Skills')
                axes[1, 0].set_title('Skills by KSAO Section')
                axes[1, 0].tick_params(axis='x', rotation=45)

            # Plot 4: Cumulative skill coverage (FIXED - handles empty data)
            sorted_counts = sorted(skill_counter.values(), reverse=True)
            if sorted_counts:
                cumsum = np.cumsum(sorted_counts)
                cumsum_pct = (cumsum / cumsum[-1]) * 100

                axes[1, 1].plot(range(1, len(cumsum_pct) + 1), cumsum_pct,
                               color='darkgreen', linewidth=2)
                axes[1, 1].axhline(80, color='red', linestyle='--', alpha=0.7,
                                   label='80% threshold')
                axes[1, 1].set_xlabel('Number of Skills (ranked by frequency)')
                axes[1, 1].set_ylabel('Cumulative % of Skill Mentions')
                axes[1, 1].set_title('Skill Coverage Curve')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No data available',
                               ha='center', va='center', fontsize=14)
                axes[1, 1].set_title('Skill Coverage Curve')

            plt.tight_layout()

            plot_path = Path(INPUT_FOLDER) / 'analysis_plots.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plots saved to: {plot_path}")
            plt.close()

        except Exception as e:
            print(f"[WARNING] Could not generate plots: {e}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
