#!/bin/bash
echo "=========================================="
echo "🔍 DEEPFAKE-GUARD FILE STRUCTURE AUDIT"
echo "=========================================="
echo ""

echo "📁 ROOT LEVEL FILES:"
ls -1 ~/deepfake-guard/ | grep -v "^d" | head -20
echo ""

echo "📂 ROOT LEVEL DIRS:"
ls -d ~/deepfake-guard/*/ 2>/dev/null | sed 's|/home/saumya/deepfake-guard/||'
echo ""

echo "=========================================="
echo "🔍 CHECKING CRITICAL FILES FROM README"
echo "=========================================="

files=(
  "backend/app/ml/models/meso4.py:Meso4 Model"
  "backend/app/ml/models/train_baseline.py:Meso4 Training"
  "backend/app/ml/ensemble/efficientnet_model.py:EfficientNet Architecture"
  "backend/app/ml/ensemble/train_efficientnet.py:EfficientNet Training"
  "backend/app/ml/ensemble/train_efficientnet_fast.py:EfficientNet Fast Training"
  "backend/app/ml/adversarial/test_fgsm_baseline.py:FGSM Attack"
  "backend/app/ml/adversarial/test_pgd_baseline.py:PGD Attack"
  "backend/app/ml/adversarial/train_robust_final.py:Adversarial Training"
  "backend/app/ml/preprocessing/extract_faces_mtcnn.py:MTCNN Extraction"
  "backend/app/ml/preprocessing/generate_test_videos.py:Test Video Generator"
  "backend/app/ml/data/download_celebdf.py:Celeb-DF Downloader"
  "backend/app/ml/data/organize_celebdf.py:Dataset Organizer"
  "data/models/meso4_baseline_best.pth:Meso4 Model Weights"
  "data/models/meso4_pgd_hardened.pth:PGD Hardened Weights"
  "data/models/efficientnet_b3_final.pth:EfficientNet Weights"
  "docs/day5.md:Day 5 Documentation"
  "docker-compose.yml:Docker Config"
  "requirements.txt:Python Requirements"
)

present=0
missing=0

for item in "${files[@]}"; do
  file="${item%%:*}"
  desc="${item##*:}"
  
  if [ -f "$HOME/deepfake-guard/$file" ]; then
    size=$(ls -lh "$HOME/deepfake-guard/$file" | awk '{print $5}')
    echo "✅ $desc"
    echo "   $file ($size)"
    ((present++))
  else
    echo "❌ $desc"
    echo "   $file (MISSING)"
    ((missing++))
  fi
  echo ""
done

echo "=========================================="
echo "📊 SUMMARY"
echo "=========================================="
echo "✅ Present: $present"
echo "❌ Missing: $missing"
echo ""

if [ $missing -eq 0 ]; then
  echo "🎉 All critical files present!"
else
  echo "⚠️  $missing files need to be created/updated"
fi

echo ""
echo "=========================================="
echo "📂 ACTUAL DIRECTORY TREE (Depth 3)"
echo "=========================================="
tree -L 3 ~/deepfake-guard/ 2>/dev/null || find ~/deepfake-guard -maxdepth 3 -type d | head -30
echo ""

echo "=========================================="
echo "📋 EXTRA FILES NOT IN README"
echo "=========================================="
find ~/deepfake-guard -maxdepth 3 -type f -name "*.py" -o -name "*.pth" -o -name "*.md" -o -name "*.yml" 2>/dev/null | grep -v __pycache__ | sort
