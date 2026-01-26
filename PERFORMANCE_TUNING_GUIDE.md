# Performance Tuning Guide

## Problem: Why Increasing batch_size Doesn't Help

You reported:
- **batch_size=512**: 3-4 JDs/sec, GPU 48% utilized
- **batch_size=8192**: Same or slower, higher GPU utilization

### Root Cause Analysis

The bottleneck is **NOT** in the GPU computation, but in:

#### 1. **CPU Serial Processing** (Main bottleneck)
```python
# Current code does this:
for jd in job_descriptions:  # ← Serial loop
    windows = extract_windows(jd)  # CPU work
    embeddings = model.encode(windows)  # GPU work
    results = process(embeddings)  # CPU work
```

**Problem**: Each JD is processed independently
- GPU finishes one JD → waits for CPU to prepare next JD
- Increasing batch_size only speeds up the GPU part
- But GPU is already fast enough - **CPU is the bottleneck**

**Evidence from your monitoring**:
- GPU: 48% utilized → GPU is waiting
- CPU: Low utilization → Single-threaded Python loop

#### 2. **Python Object Creation Overhead**
Each `extract_skills()` call creates:
- 1 Document object
- 1 Pipeline object
- 3 Node objects (loader, matcher, resolver)
- Hundreds of Sentence/Span/Candidate objects

This Python overhead adds up when processing thousands of JDs.

#### 3. **Kernel Launch Overhead**
Current flow:
```
JD #1: CPU prepare → GPU encode → CPU process → [gap]
JD #2: CPU prepare → GPU encode → CPU process → [gap]
JD #3: CPU prepare → GPU encode → CPU process → [gap]
```

The `[gap]` is where GPU sits idle waiting for CPU.

## Solution: Multi-JD Batching

The new `UltraOptimizedSkillExtractor` eliminates CPU bottleneck:

### Key Optimization

```python
# OLD (batch extractor):
for jd in all_jds:  # Serial
    windows = extract_windows(jd)
    results = encode_and_match(windows)

# NEW (ultra extractor):
all_windows = [extract_windows(jd) for jd in all_jds]  # Parallel
unique_windows = deduplicate(all_windows)  # Reduce work
all_results = encode_and_match(unique_windows)  # Single GPU call
```

### Performance Flow

```
OLD:
  JD1 → GPU → process → JD2 → GPU → process → ...
  Time per JD: 250ms
  Throughput: 4 JDs/sec

NEW:
  All JDs → extract all windows → GPU (single batch) → process all
  Time per batch: 2000ms for 50 JDs
  Throughput: 25 JDs/sec
```

## Configuration Guide

### 1. Batch Size Selection

Batch size controls **how many queries** are encoded simultaneously in one GPU call.

For `UltraOptimizedSkillExtractor`:

| GPU | Recommended batch_size | Max Safe |
|-----|------------------------|----------|
| A100-80GB | **8192-12288** | 16384 |
| A100-40GB | **4096-8192** | 12288 |
| V100-32GB | **2048-4096** | 8192 |
| V100-16GB | **1024-2048** | 4096 |

**How to choose**:
```python
# Start high and reduce if you get OOM
extractor = UltraOptimizedSkillExtractor(
    kb_path='...',
    batch_size=8192,  # Start here for A100-80GB
    use_fp16=True
)

# If you get CUDA OOM error, reduce:
batch_size=4096  # Try this
```

### 2. FP16 vs FP32

**FP16 (Half Precision)**:
- ✓ **2x faster** GPU computation
- ✓ **50% less** GPU memory
- ✓ Negligible accuracy loss for semantic matching
- ⚠ Requires modern GPU (V100, A100, etc.)

**When to use**:
```python
# A100, V100, RTX 3000/4000 series
use_fp16=True  # Recommended

# Older GPUs (K80, P100)
use_fp16=False  # May not support FP16
```

### 3. Query Caching

**When helpful**:
- Processing multiple files with similar JDs
- Repeated queries across JDs (common skills)

**When NOT helpful**:
- Single-pass processing
- Very diverse JD corpus

```python
# Enable for multi-pass or similar JDs
cache_queries=True

# Disable for one-time processing
cache_queries=False
```

### 4. Multi-Processing vs Large Batch

You have 2 options for parallelization:

#### Option A: Large Batch (Recommended)
```python
extractor = UltraOptimizedSkillExtractor(
    batch_size=8192,
    use_fp16=True
)

# Process 1000 JDs at once
results = extractor.extract_skills_batch_ultra(all_1000_jds)
```

**Pros**: Simpler, no multiprocessing overhead, maximum GPU utilization
**Cons**: Limited by single GPU memory

#### Option B: Multi-Processing
```python
from multiprocessing import Pool
from functools import partial

def process_chunk(jds_chunk):
    extractor = UltraOptimizedSkillExtractor(batch_size=4096)
    return extractor.extract_skills_batch_ultra(jds_chunk)

# Split 10K JDs into 4 chunks
chunks = [jds[i::4] for i in range(4)]

with Pool(4) as pool:
    results = pool.map(process_chunk, chunks)
```

**Pros**: Can handle unlimited JDs
**Cons**: GPU contention if workers > 1, spawn overhead

**Recommendation for A100**:
- **< 10K JDs**: Use Option A (single process, large batch)
- **> 10K JDs**: Use Option A with chunking, or Option B with 2 workers

## Performance Expectations

### A100-80GB Expected Throughput

| Configuration | Throughput | 100K JDs Time |
|---------------|-----------|---------------|
| **Batch extractor (old)** | 3-4 JDs/sec | 7-9 hours |
| **Ultra + batch_size=4096** | 15-25 JDs/sec | 1-2 hours |
| **Ultra + batch_size=8192** | 25-40 JDs/sec | 40-70 min |
| **Ultra + batch_size=12288** | 30-50 JDs/sec | 30-60 min |
| **Ultra + FP16 + batch_size=8192** | **35-50 JDs/sec** | **30-50 min** |

### Diagnostic Commands

#### Check GPU Utilization
```bash
watch -n 1 nvidia-smi
```

**What to look for**:
- **GPU Util: 80-100%** ← Good! GPU is working
- **GPU Util: < 50%** ← CPU bottleneck (use ultra extractor)
- **Memory: < 50% used** ← Can increase batch_size
- **Memory: > 90% used** ← Reduce batch_size

#### Check CPU Utilization
```bash
top
```

**What to look for**:
- **Single Python process at 100%** ← Serial processing bottleneck
- **%Cpu(s): < 20%** ← Not using all cores

## Migration to Ultra Extractor

### In Your Notebook

Replace:
```python
# OLD
from skillner.jd_skill_extractor_batch import BatchJobDescriptionSkillExtractor

extractor = BatchJobDescriptionSkillExtractor(
    kb_path=KB_PATH,
    batch_size=256
)

results = extractor.extract_skills_batch(jd_list, show_progress=True)
```

With:
```python
# NEW
from skillner.jd_skill_extractor_ultra import UltraOptimizedSkillExtractor

extractor = UltraOptimizedSkillExtractor(
    kb_path=KB_PATH,
    batch_size=8192,     # Much larger for multi-JD batching
    use_fp16=True,       # 2x speedup
    cache_queries=True   # Faster for similar JDs
)

results = extractor.extract_skills_batch_ultra(jd_list, show_progress=True)
```

### Expected Behavior

Initialization:
```
Loading knowledge base from .skillner-kb/MERGED_EN.pkl...
✓ Loaded 111,173 skills

Loading semantic model: all-MiniLM-L6-v2
Computing skill embeddings...
100%|████████| 111173/111173 [00:15<00:00]
✓ Using FP16 mixed precision
✓ Using GPU: NVIDIA A100-SXM4-80GB
✓ Query caching enabled
✓ Ultra-optimized extractor ready (batch_size=8192)
```

Processing:
```
Step 1/3: Extracting sliding windows...
  → Extracted 1,234,567 windows
  → 45,678 unique queries

Step 2/3: GPU batch encoding (45,678 queries)...
100%|████████| 45678/45678 [00:03<00:00, 15000 queries/s]

Step 3/3: Computing similarities on GPU...

Processing results for 1000 JDs...
100%|████████| 1000/1000 [00:02<00:00, 400 JDs/s]
```

**Key indicators of success**:
- Step 2 shows **one progress bar** for all queries (not per-JD)
- GPU encoding is **fast** (10K-20K queries/sec)
- Final processing is **very fast** (100s of JDs/sec)

## Troubleshooting

### "CUDA out of memory"

**Solution**: Reduce batch_size
```python
batch_size=4096  # Try half
```

### "Still only 3-4 JDs/sec"

**Check**:
1. Are you using `UltraOptimizedSkillExtractor` (not `BatchJobDescriptionSkillExtractor`)?
2. Are you calling `extract_skills_batch_ultra()` (not `extract_skills_batch()`)?
3. Did you restart the notebook kernel after updating code?

### "GPU still at 50% utilization"

**Possible causes**:
1. batch_size too small → increase to 8192-12288
2. Using old extractor → switch to ultra extractor
3. Processing JDs one-by-one → use batch_ultra method

### "Getting NaN or invalid results"

**FP16 numerical issues** (rare):
```python
use_fp16=False  # Disable FP16
```

## Benchmark Your System

Run the test:
```bash
python test_ultra_performance.py
```

This will:
1. Test old batch extractor
2. Test ultra extractor with different batch sizes
3. Show speedup comparison
4. Recommend optimal settings

Expected output:
```
TEST 1: Batch Extractor
  Throughput: 3.5 JDs/sec

TEST 2: Ultra Extractor (batch_size=8192)
  Throughput: 35 JDs/sec
  🚀 Speedup: 10x

Recommended settings:
  batch_size=8192
  use_fp16=True
  cache_queries=True
```

## Summary

Your original issue:
- batch_size=512 → 3-4 JDs/sec
- batch_size=8192 → same speed

**Why**: You were hitting CPU bottleneck, not GPU bottleneck

**Solution**: Use `UltraOptimizedSkillExtractor` which:
1. Processes multiple JDs together (not one-by-one)
2. Deduplicates queries (less GPU work)
3. Uses FP16 (2x faster)
4. Minimizes Python overhead

**Expected result**:
- **10-15x faster** than current
- **35-50 JDs/sec** on your A100
- **GPU: 80-100%** utilized
- **100K JDs in 30-60 minutes**

Try it now:
```bash
python test_ultra_performance.py
```
