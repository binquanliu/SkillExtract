#!/bin/bash

# =============================================================================
# JD Processing Setup Script
# =============================================================================
# This script helps you set up the environment for processing JD parquet files
#
# Usage:
#   chmod +x setup_jd_processing.sh
#   ./setup_jd_processing.sh
# =============================================================================

echo "=========================================================================="
echo "JD Processing Environment Setup"
echo "=========================================================================="
echo ""

# 1. Create JD directory if it doesn't exist
echo "[1/5] Creating JD directory..."
if [ ! -d "../JD" ]; then
    mkdir -p ../JD
    echo "✓ Created ../JD directory"
else
    echo "✓ ../JD directory already exists"
fi

# Count parquet files in JD directory
PARQUET_COUNT=$(find ../JD -name "*.parquet" 2>/dev/null | wc -l)
echo "  Found $PARQUET_COUNT parquet files in ../JD/"
echo ""

# 2. Create output directory
echo "[2/5] Creating output directory..."
mkdir -p output/extracted_skills
echo "✓ Created output/extracted_skills directory"
echo ""

# 3. Check Python dependencies
echo "[3/5] Checking Python dependencies..."

REQUIRED_PACKAGES=("pandas" "numpy" "tqdm" "skillner" "pyarrow" "psutil")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    python3 -c "import $package" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ $package installed"
    else
        echo "  ✗ $package NOT installed"
        MISSING_PACKAGES+=("$package")
    fi
done
echo ""

# 4. Install missing packages if any
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "[4/5] Installing missing packages..."
    echo "  Missing: ${MISSING_PACKAGES[*]}"
    read -p "  Install now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install "${MISSING_PACKAGES[@]}"
        echo "✓ Packages installed"
    else
        echo "⚠ Skipped package installation"
        echo "  You can install later with: pip install ${MISSING_PACKAGES[*]}"
    fi
else
    echo "[4/5] All required packages are installed ✓"
fi
echo ""

# 5. Check JD files and provide recommendations
echo "[5/5] Analyzing data and providing recommendations..."
echo ""

if [ $PARQUET_COUNT -eq 0 ]; then
    echo "⚠ WARNING: No parquet files found in ../JD/"
    echo ""
    echo "Please move your parquet files to ../JD/ directory:"
    echo "  Example:"
    echo "    mv /path/to/your/jd_files/*.parquet ../JD/"
    echo ""
elif [ $PARQUET_COUNT -lt 10 ]; then
    echo "✓ Found $PARQUET_COUNT parquet files"
    echo ""
    echo "Recommended configuration for small dataset:"
    echo "  BATCH_SIZE = 5"
    echo "  ROWS_PER_CHUNK = 1000"
    echo ""
else
    echo "✓ Found $PARQUET_COUNT parquet files"
    echo ""

    # Get total size of parquet files
    TOTAL_SIZE=$(du -sh ../JD 2>/dev/null | cut -f1)
    echo "  Total size: $TOTAL_SIZE"
    echo ""

    # Provide recommendations based on count
    if [ $PARQUET_COUNT -lt 50 ]; then
        echo "Recommended configuration for medium dataset:"
        echo "  BATCH_SIZE = 10"
        echo "  ROWS_PER_CHUNK = 1000"
    elif [ $PARQUET_COUNT -lt 150 ]; then
        echo "Recommended configuration for large dataset:"
        echo "  BATCH_SIZE = 10"
        echo "  ROWS_PER_CHUNK = 500"
        echo "  Consider using a machine with 16GB+ RAM"
    else
        echo "Recommended configuration for very large dataset:"
        echo "  BATCH_SIZE = 5"
        echo "  ROWS_PER_CHUNK = 500"
        echo "  Consider using a machine with 32GB+ RAM"
        echo "  Or process on cloud (AWS, Google Colab, etc.)"
    fi
fi
echo ""

# Summary
echo "=========================================================================="
echo "Setup Complete!"
echo "=========================================================================="
echo ""
echo "Directory Structure:"
echo "  ../JD/                           ← Input parquet files ($PARQUET_COUNT files)"
echo "  ./output/extracted_skills/       ← Output directory (created)"
echo "  ./extract_skills_from_jd.ipynb   ← Processing notebook"
echo ""
echo "Next Steps:"
echo "  1. Open extract_skills_from_jd.ipynb in Jupyter"
echo "  2. Adjust configuration parameters if needed"
echo "  3. Run all cells to start processing"
echo ""
echo "For detailed instructions, see: JD_PROCESSING_GUIDE.md"
echo ""

# Show first few files as example
if [ $PARQUET_COUNT -gt 0 ]; then
    echo "Sample files in ../JD/:"
    find ../JD -name "*.parquet" 2>/dev/null | head -5 | while read file; do
        SIZE=$(du -h "$file" | cut -f1)
        BASENAME=$(basename "$file")
        echo "  - $BASENAME ($SIZE)"
    done

    if [ $PARQUET_COUNT -gt 5 ]; then
        echo "  ... and $((PARQUET_COUNT - 5)) more files"
    fi
    echo ""
fi

echo "=========================================================================="
