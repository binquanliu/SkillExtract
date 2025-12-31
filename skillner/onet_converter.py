"""
ONET to SkillNER Converter - Jupyter-friendly module

This module provides easy-to-use functions for converting ONET data
and extracting skills, designed for use in Jupyter notebooks.

Example usage in Jupyter:
    from skillner.onet_converter import extract_and_save, load_and_extract_skills

    # Convert ONET JSON to knowledge base
    extract_and_save(
        input_json='my_onet.json',
        output_pkl='.skillner-kb/ONET_EN.pkl'
    )

    # Extract skills from text
    skills = load_and_extract_skills(
        text='I have experience with Microsoft Excel',
        kb_path='.skillner-kb/ONET_EN.pkl'
    )
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Union


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_skill_name(text: str) -> str:
    """
    Extract skill name from ONET formatted text.

    Args:
        text: ONET skill description (e.g., "Skill Name— Description...")

    Returns:
        Skill name (e.g., "Skill Name")
    """
    if '—' in text:
        return text.split('—', 1)[0].strip()
    elif '–' in text:
        return text.split('–', 1)[0].strip()
    return text.strip()


def extract_technology_list(text: str) -> List[str]:
    """
    Extract list of technologies from ONET tech skill text.

    Args:
        text: Technology skill text (e.g., "Software— Tool1; Tool2; Tool3")

    Returns:
        List of technology names
    """
    skills = []
    if '—' in text:
        parts = text.split('—', 1)
        if len(parts) > 1:
            items = parts[1].split(';')
            for item in items:
                item = re.sub(r'\d+\s*more\s*$', '', item)
                item = item.strip()
                if item:
                    skills.append(item)
    return skills


def extract_from_occupation(
    occupation_data: dict,
    sections: Optional[List[str]] = None
) -> Dict[str, List[Dict]]:
    """
    Extract skills from a single ONET occupation record.

    Args:
        occupation_data: Single ONET occupation dictionary
        sections: List of section names to process (default: Skills, Tech Skills, etc.)

    Returns:
        Knowledge base dictionary in SkillNER format
    """
    if sections is None:
        sections = ['Skills', 'Technology Skills', 'Work Activities',
                   'Detailed Work Activities']

    kb = {}
    occ_code = occupation_data.get('code', 'unknown')
    occ_title = occupation_data.get('title', 'unknown')

    for section_name in sections:
        section_data = occupation_data.get('sections', {}).get(section_name)
        if not section_data or section_data.get('type') != 'list':
            continue

        items = section_data.get('items', [])

        for idx, item in enumerate(items):
            extracted_skills = []

            # Extract based on section type
            if section_name == 'Technology Skills':
                # Extract category name
                skill_name = extract_skill_name(item)
                extracted_skills.append(skill_name)
                # Extract individual technologies
                tech_items = extract_technology_list(item)
                extracted_skills.extend(tech_items)
            else:
                # Extract skill name before delimiter
                skill_name = extract_skill_name(item)
                extracted_skills.append(skill_name)

            # Add to knowledge base
            for skill in extracted_skills:
                if not skill or len(skill) < 2:
                    continue

                skill_key = skill.lower()
                skill_entry = {
                    'concept_id': f'onet:{occ_code}:{section_name}:{idx}:{skill}',
                    'pref_label': skill,
                    'occupation_code': occ_code,
                    'occupation_title': occ_title,
                    'section': section_name,
                    'original_text': item
                }

                if skill_key in kb:
                    kb[skill_key].append(skill_entry)
                else:
                    kb[skill_key] = [skill_entry]

    return kb


def merge_knowledge_bases(kb_list: List[Dict]) -> Dict:
    """
    Merge multiple knowledge bases into one.

    Args:
        kb_list: List of knowledge base dictionaries

    Returns:
        Merged knowledge base
    """
    merged = {}
    for kb in kb_list:
        for skill_name, entries in kb.items():
            if skill_name in merged:
                merged[skill_name].extend(entries)
            else:
                merged[skill_name] = entries.copy()
    return merged


# =============================================================================
# High-Level API
# =============================================================================

def extract_and_save(
    input_json: Union[str, Path],
    output_pkl: Union[str, Path],
    sections: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Extract skills from ONET JSON and save to pickle file.

    This is the main function for converting ONET data in Jupyter notebooks.

    Args:
        input_json: Path to ONET JSON file
        output_pkl: Path to save the knowledge base pickle
        sections: List of sections to process (default: all main sections)
        verbose: Print progress information

    Returns:
        The generated knowledge base dictionary

    Example:
        >>> kb = extract_and_save('onet.json', '.skillner-kb/ONET_EN.pkl')
        Reading ONET data from onet.json...
        Processing 150 occupations...
        ✓ Chief Executives: 25 skills
        ✓ Software Developers: 30 skills
        ...
        Total: 5,432 unique skills
        Saved to .skillner-kb/ONET_EN.pkl
    """
    if verbose:
        print(f"Reading ONET data from {input_json}...")

    # Load JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        onet_data = json.load(f)

    # Handle both single occupation and list
    if isinstance(onet_data, dict):
        onet_data = [onet_data]

    if verbose:
        print(f"Processing {len(onet_data)} occupation(s)...\n")

    # Extract from each occupation
    kb_list = []
    for occupation in onet_data:
        kb = extract_from_occupation(occupation, sections)
        kb_list.append(kb)

        if verbose:
            title = occupation.get('title', 'Unknown')
            print(f"  ✓ {title}: {len(kb)} skills")

    # Merge all knowledge bases
    final_kb = merge_knowledge_bases(kb_list)

    if verbose:
        print(f"\nTotal: {len(final_kb)} unique skills")

    # Save to pickle
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(final_kb, f)

    if verbose:
        print(f"✓ Saved to {output_pkl}")

    return final_kb


def load_kb(kb_path: Union[str, Path]) -> Dict:
    """
    Load a knowledge base from pickle file.

    Args:
        kb_path: Path to knowledge base pickle file

    Returns:
        Knowledge base dictionary
    """
    with open(kb_path, 'rb') as f:
        return pickle.load(f)


def save_kb(kb: Dict, output_path: Union[str, Path]) -> None:
    """
    Save a knowledge base to pickle file.

    Args:
        kb: Knowledge base dictionary
        output_path: Path to save the pickle file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(kb, f)


def load_and_extract_skills(
    text: str,
    kb_path: Union[str, Path],
    max_window_size: int = 5,
    return_details: bool = False
) -> Union[List[str], List[Dict]]:
    """
    Load knowledge base and extract skills from text in one step.

    This is a convenience function for quick skill extraction in Jupyter.

    Args:
        text: Text to extract skills from
        kb_path: Path to knowledge base pickle file
        max_window_size: Maximum number of words in skill name
        return_details: If True, return detailed info; if False, just skill names

    Returns:
        List of extracted skills (strings or dicts based on return_details)

    Example:
        >>> skills = load_and_extract_skills(
        ...     text='I have skills in Microsoft Excel and problem solving',
        ...     kb_path='.skillner-kb/ONET_EN.pkl'
        ... )
        >>> print(skills)
        ['Microsoft Excel', 'problem solving']
    """
    from skillner.core import Pipeline, Document
    from skillner.text_loaders import StrTextLoader
    from skillner.matchers import SlidingWindowMatcher
    from skillner.conflict_resolvers import SpanProcessor

    # Load knowledge base
    kb = load_kb(kb_path)

    # Define query method
    def query_method(query: str):
        return kb.get(query.lower(), [])

    # Create pipeline
    doc = Document()
    pipeline = Pipeline()

    pipeline.add_node(StrTextLoader(text), name='loader')
    pipeline.add_node(
        SlidingWindowMatcher(
            query_method,
            max_window_size=max_window_size,
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

    # Run extraction
    pipeline.run(doc)

    # Collect results
    results = []
    for sentence in doc:
        for span in sentence.li_spans:
            candidate = span.metadata.get('max_candidate')
            if candidate:
                skill_text = ' '.join(sentence[candidate.window])

                if return_details:
                    skill_info = candidate.metadata
                    results.append({
                        'skill': skill_text,
                        'section': skill_info.get('section'),
                        'occupation': skill_info.get('occupation_title'),
                        'concept_id': skill_info.get('concept_id')
                    })
                else:
                    results.append(skill_text)

    return results


def extract_skills_batch(
    texts: List[str],
    kb_path: Union[str, Path],
    max_window_size: int = 5
) -> List[List[str]]:
    """
    Extract skills from multiple texts in batch.

    Args:
        texts: List of texts to process
        kb_path: Path to knowledge base
        max_window_size: Maximum skill name length

    Returns:
        List of skill lists, one per input text

    Example:
        >>> resumes = ['Resume 1...', 'Resume 2...', 'Resume 3...']
        >>> skills = extract_skills_batch(resumes, '.skillner-kb/ONET_EN.pkl')
        >>> for i, resume_skills in enumerate(skills):
        ...     print(f"Resume {i+1}: {resume_skills}")
    """
    results = []
    for text in texts:
        skills = load_and_extract_skills(text, kb_path, max_window_size)
        results.append(skills)
    return results


def merge_kb_files(
    kb_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    verbose: bool = True
) -> Dict:
    """
    Merge multiple knowledge base files into one.

    Args:
        kb_paths: List of paths to knowledge base files
        output_path: Path to save merged knowledge base
        verbose: Print progress information

    Returns:
        Merged knowledge base

    Example:
        >>> merged = merge_kb_files(
        ...     ['.skillner-kb/ESCO_EN.pkl', '.skillner-kb/ONET_EN.pkl'],
        ...     '.skillner-kb/MERGED_EN.pkl'
        ... )
    """
    kb_list = []

    for path in kb_paths:
        kb = load_kb(path)
        kb_list.append(kb)

        if verbose:
            print(f"✓ Loaded {path}: {len(kb)} skills")

    merged = merge_knowledge_bases(kb_list)

    if verbose:
        print(f"\nMerged: {len(merged)} unique skills")

    save_kb(merged, output_path)

    if verbose:
        print(f"✓ Saved to {output_path}")

    return merged


# =============================================================================
# Utility Functions
# =============================================================================

def inspect_kb(kb_path: Union[str, Path], n_samples: int = 10) -> None:
    """
    Print statistics and samples from a knowledge base.

    Args:
        kb_path: Path to knowledge base file
        n_samples: Number of sample skills to display
    """
    kb = load_kb(kb_path)

    print("=" * 70)
    print(f"Knowledge Base: {kb_path}")
    print("=" * 70)
    print(f"Total unique skills: {len(kb)}")

    # Count total entries
    total_entries = sum(len(entries) for entries in kb.values())
    print(f"Total skill entries: {total_entries}")

    # Group by section
    from collections import defaultdict
    by_section = defaultdict(int)

    for skill_name, entries in kb.items():
        for entry in entries:
            by_section[entry.get('section', 'Unknown')] += 1

    print("\nBy section:")
    for section, count in sorted(by_section.items()):
        print(f"  {section}: {count}")

    # Show samples
    print(f"\nSample skills (first {n_samples}):")
    for i, (skill_name, entries) in enumerate(list(kb.items())[:n_samples], 1):
        entry = entries[0]
        print(f"\n  {i}. {entry['pref_label']}")
        print(f"     Section: {entry.get('section', 'N/A')}")
        if 'occupation_title' in entry:
            print(f"     Occupation: {entry['occupation_title']}")


def get_kb_stats(kb_path: Union[str, Path]) -> Dict:
    """
    Get statistics about a knowledge base.

    Args:
        kb_path: Path to knowledge base file

    Returns:
        Dictionary with statistics
    """
    kb = load_kb(kb_path)

    from collections import defaultdict
    by_section = defaultdict(int)

    for skill_name, entries in kb.items():
        for entry in entries:
            by_section[entry.get('section', 'Unknown')] += 1

    return {
        'total_skills': len(kb),
        'total_entries': sum(len(entries) for entries in kb.values()),
        'by_section': dict(by_section),
        'file_size_kb': Path(kb_path).stat().st_size / 1024
    }
