"""
Enhanced query methods for skill matching with fuzzy and semantic matching.

This module provides advanced matching capabilities beyond exact string matching:
- Fuzzy matching: Handle typos and variations
- Semantic matching: Understand synonyms and related concepts
- Variant expansion: Automatically generate skill variants
"""

from typing import List, Dict, Callable, Optional
from difflib import SequenceMatcher
import pickle


class FuzzyQueryMethod:
    """
    Fuzzy string matching for skill queries.

    Handles variations like:
    - "problem solving" vs "problem-solving"
    - "Excel" vs "Microsoft Excel"
    - Minor typos and variations

    Example:
        >>> kb = load_kb('.skillner-kb/ONET_EN.pkl')
        >>> query = FuzzyQueryMethod(kb, similarity_threshold=0.85)
        >>> results = query("problem-solving")  # Matches "problem solving"
    """

    def __init__(self, kb: Dict, similarity_threshold: float = 0.85):
        """
        Args:
            kb: Knowledge base dictionary
            similarity_threshold: Minimum similarity score (0-1), higher is stricter
        """
        self.kb = kb
        self.threshold = similarity_threshold
        self.all_skills = list(kb.keys())

    def __call__(self, query: str) -> List[Dict]:
        """
        Query with fuzzy matching.

        Args:
            query: Query string

        Returns:
            List of matching skill entries
        """
        query = query.lower()

        # Exact match first (fastest)
        if query in self.kb:
            return self.kb[query]

        # Fuzzy match
        best_matches = []
        for skill_key in self.all_skills:
            similarity = SequenceMatcher(None, query, skill_key).ratio()
            if similarity >= self.threshold:
                best_matches.append((skill_key, similarity))

        if not best_matches:
            return []

        # Return best match
        best_matches.sort(key=lambda x: x[1], reverse=True)
        return self.kb[best_matches[0][0]]


class SemanticQueryMethod:
    """
    Semantic similarity matching using sentence embeddings.

    Understands synonyms and related concepts:
    - "analytical thinking" -> "Critical Thinking"
    - "spreadsheet software" -> "Microsoft Excel"
    - "verbal skills" -> "Communication"

    Requires: pip install sentence-transformers

    Example:
        >>> query = SemanticQueryMethod(kb, model_name='all-MiniLM-L6-v2')
        >>> results = query("analytical skills")  # Matches "Critical Thinking"
    """

    def __init__(
        self,
        kb: Dict,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.6
    ):
        """
        Args:
            kb: Knowledge base dictionary
            model_name: Sentence transformer model name
            similarity_threshold: Minimum cosine similarity (0-1)
        """
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )

        print(f"Loading semantic model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.util = util

        # Pre-compute skill embeddings
        print("Computing skill embeddings...")
        self.skill_keys = list(kb.keys())
        skill_texts = [kb[key][0]['pref_label'] for key in self.skill_keys]
        self.skill_embeddings = self.model.encode(skill_texts, convert_to_tensor=True)
        self.kb = kb

        print(f"✓ Ready with {len(self.skill_keys)} skills")

    def __call__(self, query: str) -> List[Dict]:
        """
        Query with semantic matching.

        Args:
            query: Query string

        Returns:
            List of matching skill entries
        """
        # Compute query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute similarities
        similarities = self.util.cos_sim(query_embedding, self.skill_embeddings)[0]

        # Find best match
        max_idx = similarities.argmax().item()
        max_sim = similarities[max_idx].item()

        if max_sim < self.threshold:
            return []

        return self.kb[self.skill_keys[max_idx]]


def expand_kb_with_variants(kb: Dict, save_path: Optional[str] = None) -> Dict:
    """
    Expand knowledge base with common skill variants.

    Adds variants like:
    - "problem solving" -> "problem-solving"
    - "Microsoft Excel" -> "Excel", "MS Excel"
    - "communicating" -> "communicate", "communication"

    Args:
        kb: Original knowledge base
        save_path: Optional path to save expanded KB

    Returns:
        Expanded knowledge base

    Example:
        >>> kb = load_kb('.skillner-kb/ONET_EN.pkl')
        >>> expanded = expand_kb_with_variants(kb, '.skillner-kb/ONET_EXPANDED.pkl')
        Original: 4856 skills
        Added: 1234 variants
        Total: 6090 skills
    """
    import re

    expanded_kb = kb.copy()

    # Variant generation rules
    variant_rules = [
        # Hyphen variations
        (r'(\w+)-(\w+)', r'\1 \2'),
        (r'(\w+) (\w+)', r'\1-\2'),

        # Microsoft abbreviations
        (r'^microsoft (\w+)', r'\1'),
        (r'^microsoft (\w+)', r'ms \1'),

        # Verb forms
        (r'(\w{4,})ing$', r'\1'),
        (r'(\w{4,})ed$', r'\1'),
        (r'(\w{4,})s$', r'\1'),
    ]

    added_count = 0

    for skill_key, entries in list(kb.items()):
        variants = set()

        for pattern, replacement in variant_rules:
            try:
                variant = re.sub(pattern, replacement, skill_key)
                if variant != skill_key and len(variant) > 2:
                    variants.add(variant)
            except:
                continue

        for variant in variants:
            if variant not in expanded_kb:
                expanded_kb[variant] = entries.copy()
                added_count += 1

    print(f"Original: {len(kb)} skills")
    print(f"Added: {added_count} variants")
    print(f"Total: {len(expanded_kb)} skills")

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(expanded_kb, f)
        print(f"✓ Saved to {save_path}")

    return expanded_kb


def add_custom_synonyms(
    kb: Dict,
    synonym_map: Dict[str, List[str]],
    save_path: Optional[str] = None
) -> Dict:
    """
    Add custom synonyms to knowledge base.

    Args:
        kb: Original knowledge base
        synonym_map: Dict mapping original skill to list of synonyms
        save_path: Optional path to save expanded KB

    Returns:
        Expanded knowledge base

    Example:
        >>> synonyms = {
        ...     'Critical Thinking': ['analytical thinking', 'critical analysis'],
        ...     'Microsoft Excel': ['Excel', 'MS Excel'],
        ... }
        >>> expanded = add_custom_synonyms(kb, synonyms)
    """
    expanded_kb = kb.copy()
    added_count = 0

    for original_skill, synonyms in synonym_map.items():
        original_key = original_skill.lower()

        if original_key in kb:
            entries = kb[original_key]

            for synonym in synonyms:
                synonym_key = synonym.lower()
                if synonym_key not in expanded_kb:
                    expanded_kb[synonym_key] = entries.copy()
                    added_count += 1
                    print(f"Added: '{synonym}' -> '{original_skill}'")

    print(f"\n✓ Added {added_count} synonyms")

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(expanded_kb, f)
        print(f"✓ Saved to {save_path}")

    return expanded_kb


# Common synonym mappings for skills
COMMON_SYNONYMS = {
    'Critical Thinking': [
        'analytical thinking',
        'analytical skills',
        'critical analysis',
        'analytical reasoning'
    ],
    'Problem Solving': [
        'problem-solving',
        'solving problems',
        'troubleshooting',
        'problem resolution'
    ],
    'Microsoft Excel': [
        'Excel',
        'MS Excel',
        'Excel spreadsheet',
        'spreadsheet software'
    ],
    'Microsoft Word': [
        'Word',
        'MS Word',
        'word processing'
    ],
    'Microsoft PowerPoint': [
        'PowerPoint',
        'MS PowerPoint',
        'PPT',
        'presentation software'
    ],
    'Communication': [
        'communicating',
        'communication skills',
        'verbal communication',
        'written communication'
    ],
    'Leadership': [
        'leading',
        'leader',
        'leadership skills',
        'team leadership'
    ],
    'Teamwork': [
        'team work',
        'collaboration',
        'collaborative work',
        'working in teams'
    ],
    'Time Management': [
        'managing time',
        'time organization',
        'prioritization'
    ],
    'Project Management': [
        'managing projects',
        'project coordination',
        'PM'
    ],
}
