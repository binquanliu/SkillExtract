#!/usr/bin/env python3
"""
Example script demonstrating how to use ONET knowledge base with SkillNER.

This script shows:
1. How to load ONET knowledge base
2. How to perform skill extraction using ONET
3. How to merge ESCO and ONET knowledge bases for comprehensive extraction
"""

import pickle
from pathlib import Path

from skillner.core import Pipeline, Document
from skillner.text_loaders import StrTextLoader
from skillner.matchers import SlidingWindowMatcher
from skillner.conflict_resolvers import SpanProcessor


# ============================================================================
# Example 1: Load and inspect ONET knowledge base
# ============================================================================

def load_knowledge_base(kb_path: str):
    """Load a knowledge base from pickle file."""
    with open(kb_path, 'rb') as f:
        return pickle.load(f)


def inspect_knowledge_base(kb: dict, kb_name: str):
    """Print statistics about a knowledge base."""
    print(f"\n{'='*60}")
    print(f"Knowledge Base: {kb_name}")
    print(f"{'='*60}")
    print(f"Total unique skill terms: {len(kb)}")

    # Count total entries
    total_entries = sum(len(entries) for entries in kb.values())
    print(f"Total skill entries: {total_entries}")

    # Show some examples
    print(f"\nSample skills:")
    for i, (skill_name, entries) in enumerate(list(kb.items())[:5]):
        print(f"\n  {i+1}. '{entries[0]['pref_label']}'")
        if 'occupation_title' in entries[0]:
            print(f"     From: {entries[0]['occupation_title']}")
        if 'section' in entries[0]:
            print(f"     Section: {entries[0]['section']}")


# ============================================================================
# Example 2: Skill extraction using ONET knowledge base
# ============================================================================

def create_query_method(knowledge_base: dict):
    """
    Create a query method for the knowledge base.

    The query method takes a string and returns matching skill entries.
    """
    def query_method(query: str):
        # Normalize query (lowercase)
        normalized_query = query.lower()

        # Look up in knowledge base
        result = knowledge_base.get(normalized_query, None)

        if result is None:
            return []

        return result

    return query_method


def extract_skills_from_text(text: str, knowledge_base: dict, kb_name: str):
    """
    Extract skills from text using the provided knowledge base.

    Args:
        text: Text to extract skills from
        knowledge_base: Knowledge base dictionary
        kb_name: Name of knowledge base (for display)
    """
    print(f"\n{'='*60}")
    print(f"Skill Extraction using {kb_name}")
    print(f"{'='*60}")
    print(f"\nInput text:\n{text}\n")

    # Create query method
    query_method = create_query_method(knowledge_base)

    # Create document
    doc = Document()

    # Build extraction pipeline
    pipeline = Pipeline()

    pipeline.add_node(
        StrTextLoader(text),
        name='loader'
    )

    pipeline.add_node(
        SlidingWindowMatcher(
            query_method,
            max_window_size=5,  # Increased to catch longer skill names
            pre_filter=lambda word: word.lower()
        ),
        name='matcher'
    )

    pipeline.add_node(
        SpanProcessor(
            dict_filters={
                "max_candidate": lambda span: max(span.li_candidates, key=len)
            }
        ),
        name="conflict_resolver"
    )

    # Run extraction
    pipeline.run(doc)

    # Display results
    print("Extracted skills:")
    print("-" * 60)

    skill_count = 0
    for sentence_idx, sentence in enumerate(doc, start=1):
        if not sentence.li_spans:
            continue

        print(f"\nSentence {sentence_idx}: {str(sentence)}")

        for span in sentence.li_spans:
            max_candidate = span.metadata.get('max_candidate')
            if max_candidate:
                skill_text = " ".join(sentence[max_candidate.window])
                skill_info = max_candidate.metadata

                print(f"  ✓ '{skill_text}'")
                if 'occupation_title' in skill_info:
                    print(f"    Related to: {skill_info['occupation_title']}")
                if 'section' in skill_info:
                    print(f"    Category: {skill_info['section']}")

                skill_count += 1

    print(f"\nTotal skills found: {skill_count}")


# ============================================================================
# Example 3: Merge multiple knowledge bases
# ============================================================================

def merge_knowledge_bases(kb1: dict, kb2: dict, kb1_name: str, kb2_name: str):
    """
    Merge two knowledge bases into one.

    Args:
        kb1: First knowledge base
        kb2: Second knowledge base
        kb1_name: Name of first KB (for display)
        kb2_name: Name of second KB (for display)

    Returns:
        Merged knowledge base dictionary
    """
    print(f"\n{'='*60}")
    print(f"Merging Knowledge Bases")
    print(f"{'='*60}")
    print(f"{kb1_name}: {len(kb1)} unique skills")
    print(f"{kb2_name}: {len(kb2)} unique skills")

    merged = {}

    # Add all entries from kb1
    for skill_name, entries in kb1.items():
        merged[skill_name] = entries.copy()

    # Add entries from kb2
    overlap_count = 0
    for skill_name, entries in kb2.items():
        if skill_name in merged:
            # Skill exists in both - merge entries
            merged[skill_name].extend(entries)
            overlap_count += 1
        else:
            # New skill from kb2
            merged[skill_name] = entries.copy()

    print(f"\nMerged KB: {len(merged)} unique skills")
    print(f"Overlapping skills: {overlap_count}")
    print(f"New skills from {kb2_name}: {len(kb2) - overlap_count}")

    return merged


# ============================================================================
# Main demonstration
# ============================================================================

def main():
    """Run all examples."""

    # Sample text for extraction
    sample_text = """
    I am an experienced Chief Executive with strong skills in judgment and decision making,
    complex problem solving, and critical thinking. I have expertise in accounting software
    like QuickBooks and have extensive experience in management of financial resources.
    My background includes coordination, negotiation, and speaking to convey information effectively.
    I'm proficient with Microsoft Excel, Microsoft PowerPoint, and enterprise resource planning ERP software.
    """

    # Define knowledge base paths
    kb_dir = Path(".skillner-kb")
    onet_kb_path = kb_dir / "ONET_EN.pkl"
    esco_kb_path = kb_dir / "ESCO_EN.pkl"

    # ========================================================================
    # Example 1: Inspect ONET knowledge base
    # ========================================================================

    if onet_kb_path.exists():
        print("\n" + "="*60)
        print("EXAMPLE 1: Inspect ONET Knowledge Base")
        print("="*60)

        onet_kb = load_knowledge_base(str(onet_kb_path))
        inspect_knowledge_base(onet_kb, "ONET")

        # ====================================================================
        # Example 2: Extract skills using ONET
        # ====================================================================

        print("\n" + "="*60)
        print("EXAMPLE 2: Skill Extraction with ONET")
        print("="*60)

        extract_skills_from_text(sample_text, onet_kb, "ONET")

    else:
        print(f"\n⚠ ONET knowledge base not found at {onet_kb_path}")
        print("Please convert your ONET JSON first:")
        print("  python scripts/convert_onet_to_skillner.py your_onet.json .skillner-kb/ONET_EN.pkl")

    # ========================================================================
    # Example 3: Merge ESCO and ONET knowledge bases
    # ========================================================================

    if onet_kb_path.exists() and esco_kb_path.exists():
        print("\n" + "="*60)
        print("EXAMPLE 3: Merge ESCO and ONET Knowledge Bases")
        print("="*60)

        onet_kb = load_knowledge_base(str(onet_kb_path))
        esco_kb = load_knowledge_base(str(esco_kb_path))

        merged_kb = merge_knowledge_bases(esco_kb, onet_kb, "ESCO", "ONET")

        # Extract skills using merged KB
        print("\n" + "="*60)
        print("EXAMPLE 4: Skill Extraction with Merged KB")
        print("="*60)

        extract_skills_from_text(sample_text, merged_kb, "ESCO + ONET")

    elif esco_kb_path.exists():
        print(f"\n⚠ To see the merged KB example, you need both ESCO and ONET knowledge bases.")
        print(f"ESCO found: {esco_kb_path.exists()}")
        print(f"ONET found: {onet_kb_path.exists()}")


if __name__ == '__main__':
    main()
