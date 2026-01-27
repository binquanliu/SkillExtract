"""
Improved extractor with context-aware filtering to reduce false positives.

Key improvements:
1. Stop phrase filtering: Remove common false positive patterns
2. Context-based filtering: Analyze surrounding words
3. Section header detection: Skip title/header text
4. Preposition filtering: Skip "with X", "for X" patterns
5. Configurable strictness levels

Reduces false positives while maintaining recall.
"""

import re
from typing import List, Dict, Set
from skillner.jd_skill_extractor_improved import ImprovedBatchSkillExtractor


class ContextAwareSkillExtractor(ImprovedBatchSkillExtractor):
    """
    Skill extractor with context-aware filtering to reduce false positives.

    Addresses issues like:
    - "collaboration with management" → should NOT extract "Management"
    - "Education and Certification:" → should NOT extract "Education"
    - "reporting to management" → should NOT extract "Management"
    """

    # Prepositions that indicate the skill is NOT being required
    # e.g., "collaboration WITH management" - management is not a skill here
    PREPOSITION_PATTERNS = {
        'with', 'for', 'to', 'of', 'from', 'by', 'under', 'about'
    }

    # Verbs that indicate the following word is an object, not a skill
    VERB_PATTERNS = {
        'reporting to', 'working with', 'collaborate with', 'collaborating with',
        'interaction with', 'communicate with', 'communicating with',
        'support for', 'supporting', 'assist', 'assisting',
        'report to', 'reporting to'
    }

    # Common section headers/titles to skip
    HEADER_PATTERNS = {
        'education and certification', 'education and certifications',
        'required qualifications', 'preferred qualifications',
        'job description', 'job summary', 'about the role',
        'responsibilities', 'requirements', 'qualifications',
        'benefits', 'about us', 'about the company'
    }

    # Phrases that are never skills (too generic or common false positives)
    STOP_PHRASES = {
        'management team', 'senior management', 'upper management',
        'team members', 'team member', 'other team members',
        'business needs', 'customer needs', 'client needs',
        'as needed', 'when needed', 'if needed',
        'years experience', 'years of experience',
        'bachelor degree', 'master degree',
        'united states', 'must have', 'should have',
        'ability to', 'willingness to', 'desire to'
    }

    def __init__(
        self,
        kb_path: str,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.65,  # Slightly higher than default
        max_window_size: int = 5,
        batch_size: int = 2048,
        use_fp16: bool = True,
        cleanup_every_n: int = 100,
        filter_level: str = 'medium'  # 'low', 'medium', 'high'
    ):
        """
        Initialize context-aware extractor.

        Args:
            filter_level: Filtering strictness
                - 'low': Minimal filtering (more false positives, fewer misses)
                - 'medium': Balanced (recommended)
                - 'high': Aggressive filtering (fewer false positives, more misses)
        """
        super().__init__(
            kb_path=kb_path,
            model_name=model_name,
            similarity_threshold=similarity_threshold,
            max_window_size=max_window_size,
            batch_size=batch_size,
            use_fp16=use_fp16,
            cleanup_every_n=cleanup_every_n
        )

        self.filter_level = filter_level

        # Adjust thresholds based on filter level
        if filter_level == 'high':
            self.similarity_threshold = max(similarity_threshold, 0.70)
        elif filter_level == 'low':
            self.similarity_threshold = min(similarity_threshold, 0.60)

        print(f"✓ Context-aware filtering enabled (level: {filter_level})")
        print(f"  Similarity threshold: {self.similarity_threshold}")

    def _extract_single(
        self,
        job_description: str,
        return_details: bool = True
    ) -> Dict:
        """Extract skills with context-aware filtering."""

        # First, get raw extraction results from parent class
        result = super()._extract_single(job_description, return_details)

        if not result['details']:
            return result

        # Filter out false positives
        filtered_details = []

        for skill_info in result['details']:
            matched_text = skill_info['matched_text'].lower()
            skill_name = skill_info['skill']

            # Apply filters
            if self._should_filter_out(matched_text, job_description.lower(), skill_info):
                continue

            filtered_details.append(skill_info)

        # Rebuild result with filtered skills
        by_section = {}
        for skill_info in filtered_details:
            section = skill_info['section']
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(skill_info['skill'])

        filtered_result = {
            'skills': [s['skill'] for s in filtered_details],
            'num_skills': len(filtered_details),
            'by_section': by_section
        }

        if return_details:
            filtered_result['details'] = filtered_details

        return filtered_result

    def _should_filter_out(
        self,
        matched_text: str,
        full_text: str,
        skill_info: Dict
    ) -> bool:
        """
        Determine if a matched skill should be filtered out as false positive.

        Returns:
            True if should be filtered (is false positive)
            False if should be kept (is real skill)
        """

        # Filter 1: Stop phrases (always filter)
        if matched_text in self.STOP_PHRASES:
            return True

        # Filter 2: Section headers (always filter)
        if matched_text in self.HEADER_PATTERNS:
            return True

        # Filter 3: Header detection by punctuation
        # "Education and Certification:" or "Requirements:"
        if self._is_likely_header(matched_text, full_text):
            return True

        # Filter 4: Preposition context
        # "collaboration WITH management" → filter "management"
        if self.filter_level in ['medium', 'high']:
            if self._has_preposition_context(matched_text, full_text):
                return True

        # Filter 5: Verb-object pattern
        # "reporting TO management" → filter "management"
        if self.filter_level in ['medium', 'high']:
            if self._is_verb_object(matched_text, full_text):
                return True

        # Filter 6: Low similarity with generic terms
        # "management" with similarity 0.62 → filter if too generic
        if self.filter_level == 'high':
            if skill_info['similarity_score'] < 0.70:
                # Check if it's a generic term
                generic_terms = {'management', 'communication', 'team', 'work'}
                if matched_text in generic_terms:
                    return True

        # Keep the skill
        return False

    def _is_likely_header(self, matched_text: str, full_text: str) -> bool:
        """Detect if matched text is likely a section header."""

        # Pattern 1: Followed by colon
        # "Education and Certification:"
        pattern1 = rf'{re.escape(matched_text)}\s*:'
        if re.search(pattern1, full_text):
            return True

        # Pattern 2: At start of line with newline after
        # "\nRequirements\n"
        pattern2 = rf'\n\s*{re.escape(matched_text)}\s*\n'
        if re.search(pattern2, full_text):
            return True

        # Pattern 3: All caps or title case at line start
        # "REQUIREMENTS" or "Requirements"
        if matched_text.isupper() or matched_text.istitle():
            pattern3 = rf'^\s*{re.escape(matched_text)}'
            if re.search(pattern3, full_text, re.MULTILINE):
                return True

        return False

    def _has_preposition_context(self, matched_text: str, full_text: str) -> bool:
        """
        Check if skill appears after a preposition.

        Example: "collaboration WITH management" → True
        """

        for prep in self.PREPOSITION_PATTERNS:
            # Pattern: "prep matched_text"
            # e.g., "with management"
            pattern = rf'\b{prep}\s+{re.escape(matched_text)}\b'
            if re.search(pattern, full_text):
                return True

        return False

    def _is_verb_object(self, matched_text: str, full_text: str) -> bool:
        """
        Check if skill is an object of a verb (not a required skill).

        Example: "reporting to management" → True
        """

        for verb_phrase in self.VERB_PATTERNS:
            # Pattern: "verb_phrase matched_text"
            # e.g., "reporting to management"
            pattern = rf'{re.escape(verb_phrase)}\s+{re.escape(matched_text)}\b'
            if re.search(pattern, full_text):
                return True

        return False


# Convenience function for quick testing
def test_filtering():
    """Test the context-aware filtering."""

    test_cases = [
        # Should NOT extract "Management"
        "Frequent collaboration with management team members.",

        # Should NOT extract "Education"
        "Education and Certification:\n- Bachelor's degree required",

        # Should NOT extract "Management" (verb-object)
        "Reporting to senior management on project status.",

        # SHOULD extract "Project Management"
        "Experience with project management and leadership required.",

        # SHOULD extract "Python"
        "Strong Python programming skills needed.",

        # Should NOT extract "Communication" (with context)
        "Interface with communication team for updates.",

        # SHOULD extract "Communication Skills"
        "Excellent written and verbal communication skills."
    ]

    extractor = ContextAwareSkillExtractor(
        kb_path='.skillner-kb/MERGED_EN.pkl',
        filter_level='medium'
    )

    print("\n" + "="*70)
    print("CONTEXT-AWARE FILTERING TEST")
    print("="*70)

    for i, test_text in enumerate(test_cases, 1):
        result = extractor.extract_skills(test_text)

        print(f"\n{i}. Text: {test_text[:60]}...")
        print(f"   Extracted: {result['skills']}")

    print("\n" + "="*70)


if __name__ == '__main__':
    test_filtering()
