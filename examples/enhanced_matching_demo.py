#!/usr/bin/env python3
"""
Demonstration of enhanced matching methods for handling synonyms and variants.

This script shows how to use:
1. Fuzzy matching (for typos and variations)
2. Semantic matching (for synonyms)
3. Knowledge base expansion with variants

Run: python examples/enhanced_matching_demo.py
"""

from skillner.onet_converter import load_kb
from skillner.enhanced_matching import (
    FuzzyQueryMethod,
    SemanticQueryMethod,
    expand_kb_with_variants,
    add_custom_synonyms,
    COMMON_SYNONYMS
)
from skillner.core import Pipeline, Document
from skillner.text_loaders import StrTextLoader
from skillner.matchers import SlidingWindowMatcher
from skillner.conflict_resolvers import SpanProcessor


def demo_exact_vs_fuzzy_vs_semantic():
    """Compare exact, fuzzy, and semantic matching."""

    print("=" * 70)
    print("DEMO: Exact vs Fuzzy vs Semantic Matching")
    print("=" * 70)

    # Load knowledge base
    print("\nLoading knowledge base...")
    kb = load_kb('.skillner-kb/ONET_EN.pkl')

    # Test text with variations and synonyms
    test_text = """
    I have strong analytical thinking and problem-solving abilities.
    Proficient in Excel spreadsheets and verbal communication.
    Experience leading teams and managing projects.
    Good at troubleshooting and critical analysis.
    """

    # Create different query methods
    exact_query = lambda q: kb.get(q.lower(), [])
    fuzzy_query = FuzzyQueryMethod(kb, similarity_threshold=0.85)

    methods = {
        'Exact Match': exact_query,
        'Fuzzy Match': fuzzy_query,
    }

    # Try semantic matching if available
    try:
        semantic_query = SemanticQueryMethod(kb, similarity_threshold=0.6)
        methods['Semantic Match'] = semantic_query
    except ImportError:
        print("\n⚠ Semantic matching not available (need: pip install sentence-transformers)")

    # Test each method
    results = {}

    for method_name, query_method in methods.items():
        print(f"\n{'='*70}")
        print(f"Testing: {method_name}")
        print(f"{'='*70}")

        doc = Document()
        pipeline = Pipeline()

        pipeline.add_node(StrTextLoader(test_text), name='loader')
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

        # Collect results
        skills = []
        for sentence in doc:
            for span in sentence.li_spans:
                candidate = span.metadata.get('max_candidate')
                if candidate:
                    matched_text = ' '.join(sentence[candidate.window])
                    skill_name = candidate.metadata['pref_label']
                    skills.append((matched_text, skill_name))

        results[method_name] = skills

        print(f"\nFound {len(skills)} skills:")
        for matched, skill in skills:
            print(f"  ✓ '{matched}' -> {skill}")

    # Compare results
    print(f"\n{'='*70}")
    print("Comparison")
    print(f"{'='*70}")

    for method_name, skills in results.items():
        print(f"{method_name:20s}: {len(skills)} skills found")


def demo_kb_expansion():
    """Demonstrate knowledge base expansion with variants."""

    print("\n" + "=" * 70)
    print("DEMO: Knowledge Base Expansion")
    print("=" * 70)

    # Load original KB
    print("\nLoading original knowledge base...")
    kb = load_kb('.skillner-kb/ONET_EN.pkl')
    print(f"Original KB: {len(kb)} skills")

    # Expand with automatic variants
    print("\n1. Adding automatic variants...")
    expanded_kb = expand_kb_with_variants(kb)

    # Add custom synonyms
    print("\n2. Adding custom synonyms...")
    final_kb = add_custom_synonyms(expanded_kb, COMMON_SYNONYMS)

    print(f"\nFinal KB: {len(final_kb)} skills")
    print(f"Increase: {len(final_kb) - len(kb)} skills ({(len(final_kb)/len(kb)-1)*100:.1f}%)")

    # Test with expanded KB
    print("\n" + "=" * 70)
    print("Testing with Expanded KB")
    print("=" * 70)

    test_queries = [
        'problem-solving',      # Hyphen variant
        'analytical thinking',  # Synonym
        'Excel',               # Abbreviation
        'MS PowerPoint',       # Microsoft variant
        'communicating',       # Verb form
        'team work',          # Spacing variant
    ]

    exact_query = lambda q: final_kb.get(q.lower(), [])

    print("\nQuery results:")
    for query in test_queries:
        results = exact_query(query)
        if results:
            print(f"  ✓ '{query}' -> {results[0]['pref_label']}")
        else:
            print(f"  ✗ '{query}' -> Not found")


def demo_synonym_test():
    """Test specific synonym pairs."""

    print("\n" + "=" * 70)
    print("DEMO: Synonym Testing")
    print("=" * 70)

    kb = load_kb('.skillner-kb/ONET_EN.pkl')

    # Test pairs: (original, synonym)
    test_pairs = [
        ('critical thinking', 'analytical thinking'),
        ('problem solving', 'troubleshooting'),
        ('microsoft excel', 'excel'),
        ('communication', 'verbal communication'),
        ('leadership', 'leading teams'),
    ]

    print("\nExact matching (before expansion):")
    exact_query = lambda q: kb.get(q.lower(), [])

    for original, synonym in test_pairs:
        orig_result = exact_query(original)
        syn_result = exact_query(synonym)

        orig_status = "✓" if orig_result else "✗"
        syn_status = "✓" if syn_result else "✗"

        print(f"  {orig_status} '{original}' | {syn_status} '{synonym}'")

    # Expand KB
    print("\nExpanding knowledge base...")
    expanded_kb = add_custom_synonyms(kb, COMMON_SYNONYMS)

    print("\nExact matching (after expansion):")
    expanded_query = lambda q: expanded_kb.get(q.lower(), [])

    for original, synonym in test_pairs:
        orig_result = expanded_query(original)
        syn_result = expanded_query(synonym)

        orig_status = "✓" if orig_result else "✗"
        syn_status = "✓" if syn_result else "✗"

        print(f"  {orig_status} '{original}' | {syn_status} '{synonym}'")


def main():
    """Run all demos."""

    print("\n" + "🎯" * 35)
    print("Enhanced Matching Demonstration")
    print("🎯" * 35)

    try:
        # Demo 1: Compare matching methods
        demo_exact_vs_fuzzy_vs_semantic()

        # Demo 2: KB expansion
        demo_kb_expansion()

        # Demo 3: Synonym testing
        demo_synonym_test()

        print("\n" + "=" * 70)
        print("Summary & Recommendations")
        print("=" * 70)

        print("""
1. For QUICK IMPROVEMENT (< 5 min):
   - Use expand_kb_with_variants() to add common variants
   - Adds ~25% more skills automatically

2. For BETTER MATCHING (< 30 min):
   - Use FuzzyQueryMethod with similarity_threshold=0.85
   - Handles typos and small variations

3. For BEST RESULTS (requires setup):
   - Install: pip install sentence-transformers
   - Use SemanticQueryMethod for synonym understanding
   - Most accurate but requires ~100MB model download

Recommended workflow:
  1. Start with KB expansion (easiest)
  2. Add fuzzy matching (good balance)
  3. Consider semantic matching if needed (best quality)
        """)

    except FileNotFoundError:
        print("\n❌ Error: Knowledge base not found at .skillner-kb/ONET_EN.pkl")
        print("Please run skill extraction first:")
        print("  python scripts/extract_skills_from_sentences.py onet.json .skillner-kb/ONET_EN.pkl")


if __name__ == '__main__':
    main()
