#!/usr/bin/env python3
"""
Convert ONET JSON format to SkillNER knowledge base format.

This script converts ONET occupation data into the format required by SkillNER.
It extracts skills from multiple sections and creates a unified knowledge base.

Usage:
    python convert_onet_to_skillner.py input.json output.pkl [--sections SECTIONS]

Arguments:
    input.json: Path to ONET JSON file
    output.pkl: Path to output pickle file
    --sections: Comma-separated list of sections to extract (default: Skills,Technology Skills,Work Activities)
"""

import json
import pickle
import argparse
import re
from typing import Dict, List, Set
from pathlib import Path


def parse_technology_skill(skill_text: str) -> List[str]:
    """
    Parse technology skill text to extract individual technologies.

    Example input:
        "Accounting software— ComputerEase; Intuit QuickBooks; Sage 50"

    Returns:
        ["Accounting software", "ComputerEase", "Intuit QuickBooks", "Sage 50"]
    """
    skills = []

    # Split by '—' or '-' to separate category from items
    if '—' in skill_text:
        parts = skill_text.split('—', 1)
        category = parts[0].strip()
        skills.append(category)

        if len(parts) > 1:
            # Split items by ';' and extract technology names
            items = parts[1].split(';')
            for item in items:
                # Remove trailing numbers like "2 more"
                item = re.sub(r'\d+\s*more\s*$', '', item)
                item = item.strip()
                if item:
                    skills.append(item)
    else:
        # If no separator, use the whole text
        skills.append(skill_text.strip())

    return skills


def parse_skill_with_description(skill_text: str) -> List[str]:
    """
    Parse skill text that contains description after '—'.

    Example input:
        "Judgment and Decision Making— Considering the relative costs..."

    Returns:
        ["Judgment and Decision Making"]
    """
    skills = []

    # Split by '—' to separate skill name from description
    if '—' in skill_text:
        skill_name = skill_text.split('—', 1)[0].strip()
        skills.append(skill_name)
    else:
        skills.append(skill_text.strip())

    return skills


def extract_skills_from_onet(onet_data: Dict, sections_to_extract: List[str]) -> Dict[str, List[Dict]]:
    """
    Extract skills from ONET occupation data.

    Args:
        onet_data: Single ONET occupation record
        sections_to_extract: List of section names to extract skills from

    Returns:
        Dictionary in SkillNER format: {skill_name: [{'concept_id': ..., ...}]}
    """
    knowledge_base = {}
    occupation_code = onet_data.get('code', 'unknown')
    occupation_title = onet_data.get('title', 'unknown')

    sections = onet_data.get('sections', {})

    for section_name in sections_to_extract:
        if section_name not in sections:
            continue

        section_data = sections[section_name]

        # Skip if not a list type section
        if section_data.get('type') != 'list':
            continue

        items = section_data.get('items', [])

        for idx, item in enumerate(items):
            skill_names = []

            # Parse based on section type
            if section_name == 'Technology Skills':
                skill_names = parse_technology_skill(item)
            elif section_name in ['Skills', 'Work Activities', 'Detailed Work Activities']:
                skill_names = parse_skill_with_description(item)
            else:
                # Default: use the whole text
                skill_names = [item.strip()]

            # Add each skill to knowledge base
            for skill_name in skill_names:
                if not skill_name:
                    continue

                # Normalize skill name (lowercase for matching)
                skill_key = skill_name.lower()

                # Create skill entry
                skill_entry = {
                    'concept_id': f'onet:{occupation_code}:{section_name}:{idx}:{skill_name}',
                    'pref_label': skill_name,
                    'occupation_code': occupation_code,
                    'occupation_title': occupation_title,
                    'section': section_name,
                    'original_text': item
                }

                # Add to knowledge base (handle duplicates)
                if skill_key in knowledge_base:
                    knowledge_base[skill_key].append(skill_entry)
                else:
                    knowledge_base[skill_key] = [skill_entry]

    return knowledge_base


def merge_knowledge_bases(kb_list: List[Dict]) -> Dict:
    """
    Merge multiple knowledge bases into one.

    Handles duplicate skills by combining their entries.
    """
    merged_kb = {}

    for kb in kb_list:
        for skill_name, entries in kb.items():
            if skill_name in merged_kb:
                merged_kb[skill_name].extend(entries)
            else:
                merged_kb[skill_name] = entries.copy()

    return merged_kb


def convert_onet_to_skillner(
    input_path: str,
    output_path: str,
    sections: List[str] = None
) -> None:
    """
    Convert ONET JSON file to SkillNER pickle format.

    Args:
        input_path: Path to ONET JSON file (can be single occupation or list)
        output_path: Path to output pickle file
        sections: List of sections to extract (default: Skills, Technology Skills, Work Activities)
    """
    if sections is None:
        sections = ['Skills', 'Technology Skills', 'Work Activities', 'Detailed Work Activities']

    print(f"Reading ONET data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        onet_data = json.load(f)

    # Handle both single occupation and list of occupations
    if isinstance(onet_data, dict):
        onet_data = [onet_data]

    print(f"Processing {len(onet_data)} occupation(s)...")
    print(f"Extracting from sections: {', '.join(sections)}")

    knowledge_bases = []
    for occupation in onet_data:
        kb = extract_skills_from_onet(occupation, sections)
        knowledge_bases.append(kb)
        print(f"  - {occupation.get('title', 'Unknown')}: {len(kb)} unique skills")

    # Merge all knowledge bases
    print("\nMerging knowledge bases...")
    final_kb = merge_knowledge_bases(knowledge_bases)

    print(f"Total unique skills: {len(final_kb)}")

    # Save to pickle
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(final_kb, f)

    print("Done!")

    # Print some statistics
    print("\n=== Statistics ===")
    print(f"Total unique skill terms: {len(final_kb)}")
    total_entries = sum(len(entries) for entries in final_kb.values())
    print(f"Total skill entries: {total_entries}")

    # Show some examples
    print("\n=== Sample Skills ===")
    sample_keys = list(final_kb.keys())[:5]
    for key in sample_keys:
        entries = final_kb[key]
        print(f"\n'{entries[0]['pref_label']}' ({len(entries)} occurrence(s))")
        for entry in entries[:2]:  # Show max 2 occurrences
            print(f"  - {entry['occupation_title']} ({entry['section']})")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONET JSON to SkillNER knowledge base format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single ONET file
  python convert_onet_to_skillner.py onet_data.json .skillner-kb/ONET_EN.pkl

  # Convert with specific sections only
  python convert_onet_to_skillner.py onet_data.json output.pkl --sections "Skills,Technology Skills"
        """
    )

    parser.add_argument('input', help='Path to ONET JSON file')
    parser.add_argument('output', help='Path to output pickle file')
    parser.add_argument(
        '--sections',
        help='Comma-separated list of sections to extract (default: Skills,Technology Skills,Work Activities,Detailed Work Activities)',
        default=None
    )

    args = parser.parse_args()

    # Parse sections if provided
    sections = None
    if args.sections:
        sections = [s.strip() for s in args.sections.split(',')]

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert
    convert_onet_to_skillner(args.input, args.output, sections)


if __name__ == '__main__':
    main()
