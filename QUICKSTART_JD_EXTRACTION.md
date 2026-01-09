# Quick Start: Extract Skills from Job Descriptions

This guide shows how to extract KSAO skills from job descriptions using the complete solution we've built.

## Overview

**Goal**: Extract skills from long job descriptions (100-1000+ words) to analyze KSAO requirement changes over time.

**Solution**: Uses semantic similarity matching to handle skill variations and synonyms, with automatic deduplication.

## Files Created

### Core Implementation
- **`skillner/jd_skill_extractor.py`**: Main extractor class with semantic matching
- **`skillner/enhanced_matching.py`**: Fuzzy and semantic matching methods
- **`skillner/onet_converter.py`**: ONET knowledge base conversion utilities

### Jupyter Notebooks
- **`notebooks/extract_skills_from_jd.ipynb`**: Complete workflow for skill extraction ⭐ **START HERE**
- **`notebooks/jd_temporal_analysis.ipynb`**: Time series analysis and trends
- **`notebooks/onet_skill_extraction_demo.ipynb`**: Basic ONET extraction demo

### Scripts
- **`scripts/batch_process_jd.py`**: Command-line batch processing with checkpoints
- **`scripts/extract_skills_from_sentences.py`**: Extract keywords from ONET sentences
- **`scripts/convert_onet_to_skillner.py`**: Convert ONET JSON to SkillNER format
- **`examples/enhanced_matching_demo.py`**: Demo of matching methods

### Documentation
- **`docs/ONET_INTEGRATION.md`**: Comprehensive integration guide
- **`docs/SENTENCE_SKILL_EXTRACTION.md`**: Sentence extraction guide
- **`docs/ONET_SECTIONS_GUIDE.md`**: ONET section reference
- **`docs/JUPYTER_USAGE.md`**: Jupyter usage guide

## Quick Start (3 Steps)

### Step 1: Prepare Your Knowledge Base

If you haven't already created the ONET knowledge base:

```python
from skillner.onet_converter import extract_and_save

# Convert ONET JSON to knowledge base
extract_and_save(
    input_json='data/onet.json',
    output_pkl='.skillner-kb/ONET_EN.pkl',
    sections=['Skills', 'Abilities', 'Knowledge', 'Technology Skills']
)
```

### Step 2: Extract Skills from Job Descriptions

Open **`notebooks/extract_skills_from_jd.ipynb`** and edit these configuration variables:

```python
# Input data
INPUT_DATA = '../data/jd_sampled.parquet'  # Your sampled job descriptions

# Knowledge base
KB_PATH = '../.skillner-kb/MERGED_EN.pkl'  # Or ONET_EN.pkl

# Output
OUTPUT_PATH = '../data/jd_extracted_skills.parquet'

# Column names
JD_TEXT_COLUMN = 'job_description'
ONET_CODE_COLUMN = 'onet_code'
DATE_COLUMN = 'post_date'
```

Then run all cells. The notebook will:
1. Load your data
2. Initialize the semantic extractor
3. Extract skills from all job descriptions (with progress bar)
4. Generate statistics and visualizations
5. Save results

### Step 3: Analyze Results

The output parquet file contains:
- `skills`: List of unique skills extracted
- `num_skills`: Count of unique skills
- `by_section`: Skills organized by KSAO category
- Original columns (onet_code, post_date, etc.)

## Usage Examples

### Extract Skills from a Single Job Description

```python
from skillner.jd_skill_extractor import JobDescriptionSkillExtractor

# Initialize extractor
extractor = JobDescriptionSkillExtractor('.skillner-kb/MERGED_EN.pkl')

# Example job description
jd = """
Position Summary: The Food Service Worker is responsible for preparing
food items while providing customer service and adhering to food safety
procedures. Maintains excellent customer service and positive attitude
towards guests and co-workers.
"""

# Extract skills
results = extractor.extract_skills(jd)

print(f"Found {results['num_skills']} unique skills")
print(f"Skills: {results['skills']}")
print(f"By section: {results['by_section']}")
```

**Output:**
```
Found 12 unique skills

Skills by category:
  [Skills]: 8 skills
    - Critical Thinking
    - Customer Service
    - Service Orientation
    ... and 5 more

  [Knowledge]: 4 skills
    - Customer and Personal Service
    - Food Production
    - Public Safety and Security
    - English Language
```

### Batch Processing

```python
# Process multiple job descriptions
jd_list = [jd1, jd2, jd3, ...]

results = extractor.extract_skills_batch(jd_list, show_progress=True)

# Get statistics
stats = extractor.get_statistics(results)
print(f"Average skills per JD: {stats['skills_per_jd']['mean']:.1f}")
print(f"Top 10 skills: {stats['top_10_skills']}")
```

## Performance Estimates

- **Semantic matching**: ~2-5 seconds per job description
- **For 30,000 JDs**: ~17-42 hours
- **For 5,000 JDs**: ~3-7 hours

**Tip**: Start with a small sample (100-500 JDs) to validate results before processing the full dataset.

## Key Features

### 1. Semantic Similarity Matching
Handles variations and synonyms:
- "analytical thinking" → "Critical Thinking"
- "spreadsheet software" → "Microsoft Excel"
- "verbal skills" → "Communication"

### 2. Automatic Deduplication
If a skill is mentioned multiple times in the same job description, it's only counted once.

### 3. Section Categorization
Skills are organized into KSAO categories:
- **Skills**: Critical Thinking, Problem Solving, etc.
- **Knowledge**: Customer Service, Food Production, etc.
- **Abilities**: Deductive Reasoning, Oral Expression, etc.
- **Technology Skills**: Microsoft Excel, Word, etc.

### 4. Long Text Handling
Works with job descriptions of 100-1000+ words. The extractor uses a sliding window approach to find skills throughout the entire text.

## Troubleshooting

### Issue: Out of Memory
**Solution**: Process in smaller batches or use the command-line script:
```bash
python scripts/batch_process_jd.py \
    --input data.parquet \
    --output results.parquet \
    --batch-size 1000
```

### Issue: Too Many/Few Skills Extracted
**Solution**: Adjust the similarity threshold:
```python
# Stricter (fewer matches)
extractor = JobDescriptionSkillExtractor(kb_path, similarity_threshold=0.7)

# More lenient (more matches)
extractor = JobDescriptionSkillExtractor(kb_path, similarity_threshold=0.5)
```

### Issue: Slow Processing
**Solution**: Use fuzzy matching instead of semantic (much faster but less accurate):
```python
from skillner.enhanced_matching import FuzzyQueryMethod
from skillner.onet_converter import load_kb

kb = load_kb('.skillner-kb/MERGED_EN.pkl')
query_method = FuzzyQueryMethod(kb, similarity_threshold=0.85)
# Use in SlidingWindowMatcher
```

## Next Steps

After extracting skills:

1. **Temporal Analysis**: See `notebooks/jd_temporal_analysis.ipynb` for:
   - Skill trend analysis over time
   - Emerging and declining skills
   - KSAO category changes

2. **Occupation Comparison**: Compare skill requirements across different ONET codes

3. **Custom Analysis**: Use the extracted skills for your specific research questions

## File Structure

```
SkillExtract/
├── skillner/
│   ├── jd_skill_extractor.py      ⭐ Main extractor class
│   ├── enhanced_matching.py        Semantic/fuzzy matching
│   └── onet_converter.py           ONET utilities
├── notebooks/
│   ├── extract_skills_from_jd.ipynb    ⭐ START HERE
│   └── jd_temporal_analysis.ipynb      Time series analysis
├── scripts/
│   ├── batch_process_jd.py         Command-line batch processing
│   └── extract_skills_from_sentences.py
├── data/
│   ├── jd_sampled.parquet          Your input data
│   └── jd_extracted_skills.parquet Output results
└── .skillner-kb/
    ├── ONET_EN.pkl                 ONET knowledge base
    └── MERGED_EN.pkl               ONET + ESCO merged
```

## Example Output

From a Food Service Worker job description:

```
Extraction Results
======================================================================
Total unique skills found: 12

Skills by category:

  [Skills]: 8 skills
    - Critical Thinking
    - Customer Service
    - Service Orientation
    - Active Listening
    - Social Perceptiveness
    ... and 3 more

  [Knowledge]: 4 skills
    - Customer and Personal Service
    - Food Production
    - Public Safety and Security
    - English Language

All extracted skills:
 1. Active Listening
 2. Critical Thinking
 3. Customer and Personal Service
 4. Customer Service
 5. English Language
 6. Food Production
 7. Oral Expression
 8. Problem Sensitivity
 9. Public Safety and Security
10. Service Orientation
11. Social Perceptiveness
12. Time Management
```

## Support

For issues or questions:
1. Check the documentation in `docs/`
2. Review the example notebooks in `notebooks/`
3. See the demo scripts in `examples/`

## Summary

This solution provides a complete pipeline for extracting KSAO skills from job descriptions:

- ✅ Handles long text (100-1000+ words)
- ✅ Semantic matching (handles variations)
- ✅ Automatic deduplication
- ✅ Section categorization
- ✅ Batch processing
- ✅ Time series analysis
- ✅ Jupyter-friendly

**Start with**: `notebooks/extract_skills_from_jd.ipynb`
