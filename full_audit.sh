#!/bin/bash
echo "=========================================="
echo "🔍 COMPLETE DEEPFAKE-GUARD FILE AUDIT"
echo "=========================================="
echo ""

# Function to list files with details
list_files() {
    local dir=$1
    local label=$2
    echo "📂 $label"
    if [ -d "$dir" ]; then
        find "$dir" -type f \( -name "*.py" -o -name "*.pth" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" -o -name "*.txt" -o -name "*.sh" \) 2>/dev/null | while read file; do
            size=$(ls -lh "$file" 2>/dev/null | awk '{print $5}')
            rel_path=$(echo "$file" | sed "s|$HOME/deepfake-guard/||")
            echo "   $rel_path ($size)"
        done
    else
        echo "   ❌ Directory does not exist"
    fi
    echo ""
}

echo "=========================================="
echo "📁 ROOT LEVEL"
echo "=========================================="
ls -lh ~/deepfake-guard/ | grep -v "^d" | grep -v "total" | awk '{print $9, "(" $5 ")"}'
echo ""

echo "=========================================="
echo "🗂️  BACKEND (All Files)"
echo "=========================================="
list_files "$HOME/deepfake-guard/backend" "BACKEND"

echo "=========================================="
echo "💾 DATA (Models & Processed)"
echo "=========================================="
echo "📍 data/models/:"
ls -lh ~/deepfake-guard/data/models/ 2>/dev/null | grep -v "^d" | awk '{print $9, "(" $5 ")"}'
echo ""

echo "📍 data/processed/:"
find ~/deepfake-guard/data/processed -maxdepth 2 -type d 2>/dev/null | head -10
echo ""

echo "=========================================="
echo "🐍 PYTHON FILES ONLY (Organized)"
echo "=========================================="
echo "Models:"
find ~/deepfake-guard -name "*.py" -path "*/models/*" 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'

echo ""
echo "Ensemble:"
find ~/deepfake-guard -name "*.py" -path "*/ensemble/*" 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'

echo ""
echo "Adversarial:"
find ~/deepfake-guard -name "*.py" -path "*/adversarial/*" 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'

echo ""
echo "Preprocessing:"
find ~/deepfake-guard -name "*.py" -path "*/preprocessing/*" 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'

echo ""
echo "Data Scripts:"
find ~/deepfake-guard -name "*.py" -path "*/data/*" 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'

echo ""
echo "Root/Other:"
find ~/deepfake-guard -maxdepth 2 -name "*.py" 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'
echo ""

echo "=========================================="
echo "⚖️  MODEL WEIGHTS (.pth files)"
echo "=========================================="
find ~/deepfake-guard -name "*.pth" -o -name "*.pt" 2>/dev/null | while read file; do
    size=$(ls -lh "$file" 2>/dev/null | awk '{print $5}')
    rel_path=$(echo "$file" | sed "s|$HOME/deepfake-guard/||")
    echo "   $rel_path ($size)"
done
echo ""

echo "=========================================="
echo "📚 DOCUMENTATION (.md files)"
echo "=========================================="
find ~/deepfake-guard -name "*.md" 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'
echo ""

echo "=========================================="
echo "🗂️  ALL DIRECTORIES (Tree View)"
echo "=========================================="
find ~/deepfake-guard -type d -not -path "*/\.*" -not -path "*/__pycache__*" | sort | sed 's|/home/saumya/deepfake-guard|.|' | head -40
echo ""

echo "=========================================="
echo "🎯 SUMMARY COUNTS"
echo "=========================================="
py_count=$(find ~/deepfake-guard -name "*.py" 2>/dev/null | wc -l)
pth_count=$(find ~/deepfake-guard -name "*.pth" 2>/dev/null | wc -l)
md_count=$(find ~/deepfake-guard -name "*.md" 2>/dev/null | wc -l)
dir_count=$(find ~/deepfake-guard -type d -not -path "*/\.*" 2>/dev/null | wc -l)

echo "Python files (.py): $py_count"
echo "Model weights (.pth): $pth_count"
echo "Documentation (.md): $md_count"
echo "Total directories: $dir_count"
echo ""

echo "=========================================="
echo "✅ CHECKLIST: Mark what you actually use"
echo "=========================================="
echo "Mark with [x] what is CORRECT and USED:"
echo ""
