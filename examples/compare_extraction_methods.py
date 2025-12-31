#!/usr/bin/env python3
"""
Compare different ONET skill extraction methods.

This script demonstrates the difference between:
1. Using full sentences as skills (original)
2. Extracting skill keywords from sentences (recommended)
"""

import pickle
from pathlib import Path


def load_kb(path: str):
    """Load knowledge base from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def demo_sentence_vs_keyword():
    """
    Demonstrate why extracting keywords from sentences is important.
    """
    print("=" * 70)
    print("ONET Skill Extraction: Sentence vs Keyword Comparison")
    print("=" * 70)

    # Example resume text
    resume_text = """
    Senior Software Engineer with 8 years of experience.

    Core Competencies:
    - Strong judgment and decision making
    - Complex problem solving
    - Critical thinking and analysis
    - Proficient in Microsoft Excel and PowerPoint
    - Experience with accounting software like QuickBooks
    - Excellent coordination and communication skills
    """

    print("\n📄 Sample Resume Text:")
    print(resume_text)

    # Simulate different matching scenarios
    print("\n" + "=" * 70)
    print("Scenario 1: Using Full Sentences (NOT Recommended)")
    print("=" * 70)

    full_sentences = [
        "Judgment and Decision Making— Considering the relative costs and benefits of potential actions to choose the most appropriate one.",
        "Complex Problem Solving— Identifying complex problems and reviewing related information to develop and evaluate options and implement solutions.",
        "Spreadsheet software— Microsoft Excel; Microsoft PowerPoint",
        "Accounting software— ComputerEase construction accounting software; Fund accounting software; Intuit QuickBooks; Sage 50 Accounting"
    ]

    print("\nKnowledge Base Entries (full sentences):")
    for i, sent in enumerate(full_sentences, 1):
        print(f"  {i}. {sent}")

    print("\n❌ Matching Results:")
    print("  - 'judgment and decision making' → NOT FOUND")
    print("    (Resume has 'judgment and decision making', KB has full sentence)")
    print("  - 'Microsoft Excel' → NOT FOUND")
    print("    (Can't match partial text from 'Spreadsheet software— Microsoft Excel; ...')")
    print("  - 'QuickBooks' → NOT FOUND")
    print("    (Buried in a long list)")
    print("\n💔 Total Matches: 0/5 skills")

    # With keyword extraction
    print("\n" + "=" * 70)
    print("Scenario 2: Using Extracted Keywords (Recommended)")
    print("=" * 70)

    extracted_keywords = [
        "Judgment and Decision Making",
        "Complex Problem Solving",
        "Critical Thinking",
        "Spreadsheet software",
        "Microsoft Excel",
        "Microsoft PowerPoint",
        "Accounting software",
        "QuickBooks",
        "Sage 50 Accounting"
    ]

    print("\nKnowledge Base Entries (extracted keywords):")
    for i, keyword in enumerate(extracted_keywords, 1):
        print(f"  {i}. {keyword}")

    print("\n✅ Matching Results:")
    print("  - 'judgment and decision making' → FOUND! ✓")
    print("  - 'complex problem solving' → FOUND! ✓")
    print("  - 'critical thinking' → FOUND! ✓")
    print("  - 'Microsoft Excel' → FOUND! ✓")
    print("  - 'QuickBooks' → FOUND! ✓")
    print("\n💚 Total Matches: 5/5 skills")

    # Statistics
    print("\n" + "=" * 70)
    print("Statistics Comparison")
    print("=" * 70)

    print("\n| Metric | Full Sentences | Extracted Keywords |")
    print("|--------|---------------|-------------------|")
    print("| Entries | 4 | 9 |")
    print("| Avg Length | 85 chars | 22 chars |")
    print("| Matches | 0 | 5 |")
    print("| Match Rate | 0% | 100% |")
    print("| Storage | ~340 bytes | ~200 bytes |")

    # Show extraction examples
    print("\n" + "=" * 70)
    print("Extraction Examples")
    print("=" * 70)

    examples = [
        {
            'original': 'Judgment and Decision Making— Considering the relative costs and benefits of potential actions to choose the most appropriate one.',
            'extracted': ['Judgment and Decision Making'],
            'method': 'Pattern (before "—")'
        },
        {
            'original': 'Accounting software— ComputerEase construction accounting software; Fund accounting software; Intuit QuickBooks; Sage 50 Accounting',
            'extracted': ['Accounting software', 'ComputerEase construction accounting software',
                        'Fund accounting software', 'Intuit QuickBooks', 'Sage 50 Accounting'],
            'method': 'Pattern + List splitting'
        },
        {
            'original': 'Direct or coordinate an organization\'s financial or budget activities to fund operations, maximize investments, or increase efficiency.',
            'extracted': ['Direct financial activities', 'budget activities', 'operations', 'investments'],
            'method': 'NLP (noun phrases)'
        }
    ]

    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Original sentence:")
        print(f"  '{ex['original']}'")
        print(f"\nExtracted keywords ({ex['method']}):")
        for kw in ex['extracted']:
            print(f"  ✓ '{kw}'")


def compare_actual_files():
    """
    Compare actual generated knowledge bases if they exist.
    """
    original_kb = Path('tests/test_onet_output.pkl')
    extracted_kb = Path('tests/test_extracted_skills.pkl')

    if not original_kb.exists() or not extracted_kb.exists():
        print("\n⚠️  To compare actual files, run:")
        print("  python scripts/convert_onet_to_skillner.py tests/test_onet_sample.json tests/test_onet_output.pkl")
        print("  python scripts/extract_skills_from_sentences.py tests/test_onet_sample.json tests/test_extracted_skills.pkl")
        return

    print("\n" + "=" * 70)
    print("Actual Knowledge Base Comparison")
    print("=" * 70)

    kb1 = load_kb(str(original_kb))
    kb2 = load_kb(str(extracted_kb))

    print(f"\nOriginal (full text): {len(kb1)} unique entries")
    print(f"Extracted (keywords): {len(kb2)} unique entries")

    # Show some sample differences
    print("\nSample entries from each:")

    print("\n[Original KB - Full Text]")
    for i, (key, value) in enumerate(list(kb1.items())[:3]):
        print(f"  {i+1}. '{value[0]['pref_label'][:60]}...'")

    print("\n[Extracted KB - Keywords]")
    for i, (key, value) in enumerate(list(kb2.items())[:3]):
        print(f"  {i+1}. '{value[0]['pref_label']}'")


def main():
    """Run all comparisons."""
    demo_sentence_vs_keyword()
    compare_actual_files()

    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    print("""
✅ RECOMMENDED: Use extract_skills_from_sentences.py

Benefits:
  1. Better matching with resume/job description text
  2. More concise and standardized skill names
  3. Smaller knowledge base size
  4. Extracts multiple skills from single sentence
  5. Improved extraction accuracy

Usage:
  python scripts/extract_skills_from_sentences.py your_onet.json output.pkl

For more details:
  - See docs/SENTENCE_SKILL_EXTRACTION.md
  - See ONET_QUICKSTART.md
    """)


if __name__ == '__main__':
    main()
