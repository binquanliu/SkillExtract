#!/usr/bin/env python3
"""
Extract skill keywords from ONET sentence descriptions.

This script processes ONET data where skills are described in full sentences
and extracts the core skill terms/keywords from those sentences.

Multiple extraction methods are provided:
1. Pattern-based extraction (extract skill names before '—')
2. Noun phrase extraction using NLP
3. Keyword extraction using TF-IDF
4. Manual extraction using predefined patterns
"""

import json
import pickle
import argparse
import re
from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict, Counter


# =============================================================================
# Method 1: Pattern-based extraction (fastest, works for formatted text)
# =============================================================================

def extract_skill_name_before_dash(text: str) -> str:
    """
    Extract skill name from formatted text like:
    "Judgment and Decision Making— Considering the relative..."

    Returns: "Judgment and Decision Making"
    """
    if '—' in text:
        return text.split('—', 1)[0].strip()
    elif '–' in text:  # en dash
        return text.split('–', 1)[0].strip()
    elif '-' in text and len(text.split('-')[0]) < 100:
        # Only if the part before dash is short (likely a title)
        return text.split('-', 1)[0].strip()
    return text.strip()


def extract_technology_skills(text: str) -> List[str]:
    """
    Extract technology names from formatted text like:
    "Accounting software— QuickBooks; Sage 50; Excel"

    Returns: ["QuickBooks", "Sage 50", "Excel"]
    """
    skills = []

    if '—' in text:
        parts = text.split('—', 1)
        if len(parts) > 1:
            # Split by semicolon to get individual items
            items = parts[1].split(';')
            for item in items:
                # Remove trailing numbers like "2 more"
                item = re.sub(r'\d+\s*more\s*$', '', item)
                item = item.strip()
                if item:
                    skills.append(item)

    return skills


# =============================================================================
# Method 2: Noun Phrase Extraction (requires spaCy)
# =============================================================================

def extract_noun_phrases_spacy(text: str, nlp=None) -> List[str]:
    """
    Extract noun phrases from text using spaCy.

    This extracts meaningful noun phrases that could be skills.
    Requires: pip install spacy && python -m spacy download en_core_web_sm
    """
    try:
        import spacy
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)

        # Extract noun chunks
        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Filter out very short or very long phrases
            if 2 <= len(chunk.text) <= 50:
                noun_phrases.append(chunk.text.strip())

        return noun_phrases
    except ImportError:
        print("Warning: spaCy not installed. Install with: pip install spacy")
        print("Then: python -m spacy download en_core_web_sm")
        return []


# =============================================================================
# Method 3: Keyword Extraction using simple heuristics
# =============================================================================

def extract_keywords_simple(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords using simple heuristics:
    - Capitalized words (likely proper nouns/skills)
    - Common skill-related patterns
    """
    keywords = []

    # Remove the description part if there's a dash
    if '—' in text:
        text = text.split('—', 1)[0]

    # Find capitalized words/phrases
    # Pattern: capitalized word followed by more capitalized/lowercase words
    pattern = r'\b[A-Z][a-zA-Z]*(?:\s+(?:and|of|or|the|in|with|for|to)?\s*[A-Z]?[a-zA-Z]+)*\b'
    matches = re.findall(pattern, text)

    for match in matches:
        if len(match) >= min_length:
            keywords.append(match.strip())

    return keywords


# =============================================================================
# Method 4: Action + Object extraction (for task descriptions)
# =============================================================================

def extract_action_objects(text: str) -> List[Tuple[str, str]]:
    """
    Extract action-object pairs from task descriptions.

    Example: "Direct financial operations" -> ("Direct", "financial operations")
    """
    # Common action verbs in job descriptions
    action_verbs = {
        'direct', 'manage', 'coordinate', 'develop', 'implement', 'analyze',
        'create', 'establish', 'maintain', 'conduct', 'prepare', 'organize',
        'evaluate', 'monitor', 'provide', 'ensure', 'perform', 'review',
        'communicate', 'supervise', 'plan', 'design', 'execute', 'lead'
    }

    # Simple pattern: Verb + everything until period/comma
    words = text.lower().split()

    action_objects = []
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in action_verbs:
            # Get the object (next few words)
            obj_words = []
            for j in range(i + 1, min(i + 6, len(words))):
                w = words[j]
                # Stop at punctuation or certain words
                if w in ['to', 'in', 'by', 'with', 'and'] or ',' in w or '.' in w:
                    break
                obj_words.append(w)

            if obj_words:
                action = word.capitalize()
                obj = ' '.join(obj_words)
                obj = re.sub(r'[,.]', '', obj)  # Clean punctuation
                action_objects.append((action, obj))

    return action_objects


# =============================================================================
# Main Conversion Function
# =============================================================================

def extract_skills_from_onet(
    onet_data: Dict,
    method: str = 'auto',
    sections_to_process: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    Extract skills from ONET data using specified method.

    Args:
        onet_data: ONET occupation record
        method: Extraction method ('auto', 'pattern', 'nlp', 'keyword', 'action')
        sections_to_process: Which sections to process

    Returns:
        Dictionary in SkillNER format
    """
    if sections_to_process is None:
        sections_to_process = ['Skills', 'Abilities', 'Knowledge',
                              'Technology Skills', 'Work Activities',
                              'Detailed Work Activities', 'Tasks']

    knowledge_base = {}
    occupation_code = onet_data.get('code', 'unknown')
    occupation_title = onet_data.get('title', 'unknown')
    sections = onet_data.get('sections', {})

    # Initialize spaCy if needed
    nlp = None
    if method in ['nlp', 'auto']:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except:
            if method == 'nlp':
                print("Warning: spaCy not available, falling back to pattern method")
                method = 'pattern'

    for section_name in sections_to_process:
        if section_name not in sections:
            continue

        section_data = sections[section_name]
        if section_data.get('type') != 'list':
            continue

        items = section_data.get('items', [])

        for idx, item in enumerate(items):
            extracted_skills = []

            # Choose extraction method
            if method == 'auto':
                # Auto-detect best method based on section and content
                if section_name == 'Technology Skills':
                    # Use pattern method for technology skills
                    skill_name = extract_skill_name_before_dash(item)
                    extracted_skills.append(skill_name)
                    # Also extract individual technologies
                    tech_items = extract_technology_skills(item)
                    extracted_skills.extend(tech_items)
                elif section_name in ['Skills', 'Work Activities']:
                    # Extract the skill name before description
                    skill_name = extract_skill_name_before_dash(item)
                    extracted_skills.append(skill_name)
                elif section_name in ['Tasks', 'Detailed Work Activities']:
                    # Try NLP noun phrase extraction if available
                    if nlp:
                        noun_phrases = extract_noun_phrases_spacy(item, nlp)
                        extracted_skills.extend(noun_phrases[:3])  # Top 3
                    else:
                        # Fall back to keyword extraction
                        keywords = extract_keywords_simple(item)
                        extracted_skills.extend(keywords[:3])
                else:
                    # Default: pattern-based
                    skill_name = extract_skill_name_before_dash(item)
                    extracted_skills.append(skill_name)

            elif method == 'pattern':
                skill_name = extract_skill_name_before_dash(item)
                extracted_skills.append(skill_name)
                if section_name == 'Technology Skills':
                    extracted_skills.extend(extract_technology_skills(item))

            elif method == 'nlp':
                noun_phrases = extract_noun_phrases_spacy(item, nlp)
                extracted_skills.extend(noun_phrases)

            elif method == 'keyword':
                keywords = extract_keywords_simple(item)
                extracted_skills.extend(keywords)

            elif method == 'action':
                action_objects = extract_action_objects(item)
                for action, obj in action_objects:
                    extracted_skills.append(f"{action} {obj}")

            # Add each extracted skill to knowledge base
            for skill_name in extracted_skills:
                if not skill_name or len(skill_name) < 2:
                    continue

                # Normalize for matching
                skill_key = skill_name.lower()

                skill_entry = {
                    'concept_id': f'onet:{occupation_code}:{section_name}:{idx}:{skill_name}',
                    'pref_label': skill_name,
                    'occupation_code': occupation_code,
                    'occupation_title': occupation_title,
                    'section': section_name,
                    'original_text': item,
                    'extraction_method': method
                }

                if skill_key in knowledge_base:
                    knowledge_base[skill_key].append(skill_entry)
                else:
                    knowledge_base[skill_key] = [skill_entry]

    return knowledge_base


def merge_knowledge_bases(kb_list: List[Dict]) -> Dict:
    """Merge multiple knowledge bases."""
    merged = {}
    for kb in kb_list:
        for skill_name, entries in kb.items():
            if skill_name in merged:
                merged[skill_name].extend(entries)
            else:
                merged[skill_name] = entries.copy()
    return merged


def convert_with_skill_extraction(
    input_path: str,
    output_path: str,
    method: str = 'auto',
    sections: List[str] = None
) -> None:
    """
    Convert ONET JSON to SkillNER format with skill extraction from sentences.

    Args:
        input_path: Path to ONET JSON
        output_path: Path to output pickle
        method: Extraction method (auto, pattern, nlp, keyword, action)
        sections: Sections to process
    """
    print(f"Reading ONET data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        onet_data = json.load(f)

    if isinstance(onet_data, dict):
        onet_data = [onet_data]

    print(f"\nProcessing {len(onet_data)} occupation(s)...")
    print(f"Extraction method: {method}")
    if sections:
        print(f"Sections: {', '.join(sections)}")

    knowledge_bases = []
    for occupation in onet_data:
        kb = extract_skills_from_onet(occupation, method, sections)
        knowledge_bases.append(kb)
        print(f"  - {occupation.get('title', 'Unknown')}: {len(kb)} unique skills extracted")

    print("\nMerging knowledge bases...")
    final_kb = merge_knowledge_bases(knowledge_bases)

    print(f"Total unique skills: {len(final_kb)}")

    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(final_kb, f)

    print("Done!")

    # Statistics
    print("\n=== Statistics ===")
    print(f"Total unique skill terms: {len(final_kb)}")
    total_entries = sum(len(entries) for entries in final_kb.values())
    print(f"Total skill entries: {total_entries}")

    # Show examples
    print("\n=== Sample Extracted Skills ===")
    for i, (key, entries) in enumerate(list(final_kb.items())[:10]):
        print(f"\n{i+1}. '{entries[0]['pref_label']}'")
        print(f"   Section: {entries[0]['section']}")
        print(f"   From: {entries[0]['original_text'][:80]}...")


def main():
    parser = argparse.ArgumentParser(
        description='Extract skills from ONET sentence descriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extraction Methods:
  auto     - Automatically choose best method per section (default)
  pattern  - Extract skill names before '—' or other delimiters
  nlp      - Use spaCy NLP to extract noun phrases (requires spaCy)
  keyword  - Extract keywords using simple heuristics
  action   - Extract action-object pairs from tasks

Examples:
  # Auto method (recommended)
  python extract_skills_from_sentences.py onet.json output.pkl

  # Pattern-based extraction
  python extract_skills_from_sentences.py onet.json output.pkl --method pattern

  # NLP-based extraction (requires spaCy)
  python extract_skills_from_sentences.py onet.json output.pkl --method nlp

  # Process specific sections only
  python extract_skills_from_sentences.py onet.json output.pkl --sections "Skills,Tasks"
        """
    )

    parser.add_argument('input', help='Path to ONET JSON file')
    parser.add_argument('output', help='Path to output pickle file')
    parser.add_argument(
        '--method',
        choices=['auto', 'pattern', 'nlp', 'keyword', 'action'],
        default='auto',
        help='Skill extraction method (default: auto)'
    )
    parser.add_argument(
        '--sections',
        help='Comma-separated sections to process',
        default=None
    )

    args = parser.parse_args()

    sections = None
    if args.sections:
        sections = [s.strip() for s in args.sections.split(',')]

    convert_with_skill_extraction(args.input, args.output, args.method, sections)


if __name__ == '__main__':
    main()
