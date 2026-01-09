"""
Complete solution for extracting KSAO skills from long job descriptions.

This script handles:
- Long job descriptions (100-1000+ words)
- Semantic similarity matching (handles variations)
- Automatic deduplication
- Section-based skill categorization (Skills, Knowledge, Abilities, Technology)

Usage in Jupyter:
    from jd_skill_extractor import JobDescriptionSkillExtractor

    extractor = JobDescriptionSkillExtractor('.skillner-kb/MERGED_EN.pkl')
    results = extractor.extract_skills(job_description_text)
"""

import pandas as pd
from typing import List, Dict, Optional
from skillner.enhanced_matching import SemanticQueryMethod
from skillner.onet_converter import load_kb
from skillner.core import Pipeline, Document
from skillner.text_loaders import StrTextLoader
from skillner.matchers import SlidingWindowMatcher
from skillner.conflict_resolvers import SpanProcessor


class JobDescriptionSkillExtractor:
    """
    Extract KSAO skills from job descriptions using semantic matching.

    Features:
    - Semantic similarity matching (handles skill variations)
    - Automatic deduplication
    - Section categorization (Skills, Abilities, Knowledge, Technology)
    - Handles long text (500+ words)

    Example:
        >>> extractor = JobDescriptionSkillExtractor('.skillner-kb/MERGED_EN.pkl')
        >>> jd = "Prepares food while providing customer service..."
        >>> results = extractor.extract_skills(jd)
        >>> print(f"Found {len(results['skills'])} unique skills")
    """

    def __init__(
        self,
        kb_path: str,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.6,
        max_window_size: int = 5
    ):
        """
        Initialize the extractor.

        Args:
            kb_path: Path to knowledge base pickle file
            model_name: Sentence transformer model name
            similarity_threshold: Minimum similarity score (0-1)
            max_window_size: Maximum words in a skill phrase
        """
        print(f"Loading knowledge base from {kb_path}...")
        self.kb = load_kb(kb_path)
        print(f"✓ Loaded {len(self.kb):,} skills")

        print(f"\nLoading semantic model: {model_name}...")
        self.query_method = SemanticQueryMethod(
            self.kb,
            model_name=model_name,
            similarity_threshold=similarity_threshold
        )

        self.max_window_size = max_window_size
        print("✓ Extractor ready\n")

    def extract_skills(
        self,
        job_description: str,
        return_details: bool = True
    ) -> Dict:
        """
        Extract skills from a job description.

        Args:
            job_description: Job description text
            return_details: If True, return full details; if False, just skill names

        Returns:
            Dictionary with extracted skills:
            {
                'skills': [skill1, skill2, ...],  # Unique skills
                'num_skills': int,
                'by_section': {
                    'Skills': [...],
                    'Abilities': [...],
                    'Knowledge': [...],
                    'Technology Skills': [...]
                },
                'details': [  # If return_details=True
                    {
                        'skill': 'Critical Thinking',
                        'section': 'Skills',
                        'matched_text': 'critical thinking',
                        'concept_id': '...'
                    },
                    ...
                ]
            }
        """
        # Validate input
        if not job_description or pd.isna(job_description):
            return self._empty_result()

        job_description = str(job_description).strip()

        if len(job_description) < 10:
            return self._empty_result()

        # Build extraction pipeline
        doc = Document()
        pipeline = Pipeline()

        pipeline.add_node(
            StrTextLoader(job_description),
            name='loader'
        )

        pipeline.add_node(
            SlidingWindowMatcher(
                self.query_method,
                max_window_size=self.max_window_size,
                pre_filter=lambda w: w.lower()
            ),
            name='matcher'
        )

        pipeline.add_node(
            SpanProcessor(
                dict_filters={
                    'max_candidate': lambda span: max(span.li_candidates, key=len)
                }
            ),
            name='resolver'
        )

        # Run extraction
        pipeline.run(doc)

        # Collect and deduplicate skills
        skills_dict = {}  # key: skill_name, value: skill info

        for sentence in doc:
            for span in sentence.li_spans:
                candidate = span.metadata.get('max_candidate')
                if candidate:
                    skill_name = candidate.metadata['pref_label']

                    # Deduplicate: only keep first occurrence
                    if skill_name not in skills_dict:
                        matched_text = ' '.join(sentence[candidate.window])

                        skills_dict[skill_name] = {
                            'skill': skill_name,
                            'section': candidate.metadata.get('section', 'Unknown'),
                            'matched_text': matched_text,
                            'concept_id': candidate.metadata.get('concept_id', ''),
                            'similarity_score': candidate.metadata.get('similarity_score', 0.0)
                        }

        # Organize results
        skills_list = list(skills_dict.values())

        # Group by section
        by_section = {}
        for skill_info in skills_list:
            section = skill_info['section']
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(skill_info['skill'])

        result = {
            'skills': [s['skill'] for s in skills_list],
            'num_skills': len(skills_list),
            'by_section': by_section
        }

        if return_details:
            result['details'] = skills_list

        return result

    def extract_skills_batch(
        self,
        job_descriptions: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Extract skills from multiple job descriptions.

        Args:
            job_descriptions: List of JD texts
            show_progress: Show progress bar

        Returns:
            List of extraction results
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(job_descriptions, desc="Extracting skills")
        else:
            iterator = job_descriptions

        for jd in iterator:
            result = self.extract_skills(jd, return_details=True)
            results.append(result)

        return results

    def _empty_result(self) -> Dict:
        """Return empty result for invalid input."""
        return {
            'skills': [],
            'num_skills': 0,
            'by_section': {},
            'details': []
        }

    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Get statistics from batch extraction results.

        Args:
            results: List of extraction results from extract_skills_batch

        Returns:
            Statistics dictionary
        """
        import numpy as np

        skill_counts = [r['num_skills'] for r in results]

        # Count skills by section
        section_counts = {}
        for result in results:
            for section, skills in result['by_section'].items():
                if section not in section_counts:
                    section_counts[section] = []
                section_counts[section].append(len(skills))

        section_stats = {}
        for section, counts in section_counts.items():
            section_stats[section] = {
                'mean': np.mean(counts),
                'median': np.median(counts),
                'min': np.min(counts),
                'max': np.max(counts)
            }

        # Most common skills
        all_skills = []
        for result in results:
            all_skills.extend(result['skills'])

        from collections import Counter
        skill_freq = Counter(all_skills)

        return {
            'total_jds': len(results),
            'skills_per_jd': {
                'mean': np.mean(skill_counts),
                'median': np.median(skill_counts),
                'min': np.min(skill_counts),
                'max': np.max(skill_counts),
                'std': np.std(skill_counts)
            },
            'by_section': section_stats,
            'top_10_skills': skill_freq.most_common(10),
            'unique_skills_total': len(skill_freq)
        }


def demo_example():
    """
    Demo: Extract skills from the Food Service Worker example.
    """
    # Example job description
    jd_text = """
    Position Summary: The Food Service Worker is responsible for preparing and/or
    building food items while providing customer service and adhering to food safety,
    food handling, and sanitation procedures.

    Essential Functions:
    - Prepares and builds food items according to standardized recipes and directions
    - Properly stores food by adhering to food safety policies and procedures
    - Sets up work stations including prep tables, service counters, hot wells, steam tables, etc.
    - Breaks down, cleans, and sanitizes work stations
    - Serves food to customers while ensuring guest satisfaction and anticipating the customers' needs
    - Replenishes food items and ensure product is stocked to appropriate levels
    - Maintains excellent customer service and positive attitude towards guest, customers, clients, co-workers, etc.
    - Adheres to Aramark safety policies and procedures including proper food safety and sanitation
    - Ensures security of company assets
    - Other duties and tasks as assigned by manager
    """

    print("="*70)
    print("DEMO: Extracting skills from Food Service Worker job description")
    print("="*70)

    # Initialize extractor
    extractor = JobDescriptionSkillExtractor('.skillner-kb/MERGED_EN.pkl')

    # Extract skills
    print("\nExtracting skills...")
    results = extractor.extract_skills(jd_text)

    # Display results
    print(f"\n{'='*70}")
    print(f"Extraction Results")
    print(f"{'='*70}")
    print(f"Total unique skills found: {results['num_skills']}")

    print(f"\nSkills by category:")
    for section, skills in sorted(results['by_section'].items()):
        print(f"\n  [{section}]: {len(skills)} skills")
        for skill in skills[:5]:  # Show first 5
            print(f"    - {skill}")
        if len(skills) > 5:
            print(f"    ... and {len(skills) - 5} more")

    print(f"\n{'='*70}")
    print(f"All extracted skills:")
    print(f"{'='*70}")
    for i, skill in enumerate(sorted(results['skills']), 1):
        print(f"{i:2d}. {skill}")

    return results


if __name__ == '__main__':
    demo_example()
