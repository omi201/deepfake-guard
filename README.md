# Adversarial Deepfake Detection System (ADDS)

**Security Objective:** Build a hardened deepfake detection system resistant to adversarial evasion attacks.

---

## 🎯 Current Status

| Day | Component | Status | Key Result |
|-----|-----------|--------|------------|
| 1 | Docker Infrastructure | ✅ Complete | Postgres + Redis running |
| 2 | Dataset Acquisition | ✅ Complete | 140k + 61k Celeb-DF extracted |
| 3 | MTCNN Extraction | ✅ Complete | 61,855 faces extracted |
| 4 | Meso4 Baseline | ✅ Complete | 80% accuracy, 7.3k params |
| 5 | EfficientNet-B3 | ✅ Complete | **99.71% accuracy** |
| 6 | Data Augmentation | 🔄 Next | Albumentations pipeline |
| 7 | Ensemble Training | ⏳ Planned | Target 92% |

**Git:** `77b5dd6` - Day 5 COMPLETE: 99.71% accuracy achieved

---

## 📊 Datasets

| Dataset | Images | Split | Location |
|---------|--------|-------|----------|
| 140k Real vs Fake | 140,000 | 100k/20k/20k | `data/processed/140k/` |
| Celeb-DF v2 | 61,855 | 49.5k/12.4k | `data/processed/celeb-df/` |
| **Combined** | **182,255** | 149.5k/32.4k | ConcatDataset |

**Extraction:** MTCNN, 224x224 RGB, 72 min processing

---

## 🤖 Models

| Model | Arch | Accuracy | Params | File |
|-------|------|----------|--------|------|
| Meso4 Baseline | Custom CNN | 80% | 7.3k | `meso4_baseline_best.pth` |
| Meso4 PGD | Adv Trained | 50% robust | 7.3k | `meso4_pgd_hardened.pth` |
| **EfficientNet-B3** | **Transfer Learning** | **99.71%** | **11.4M** | `efficientnet_b3_final.pth` |

**Training Phases:**
1. Frozen backbone: 86.72%
2. Partial unfreeze: 99.44%
3. Full fine-tune: **99.71%**

---

## 🚀 Quick Start


# Activate environment
conda activate deepfake-guard
export LD_LIBRARY_PATH=/home/saumya/yes/lib:$LD_LIBRARY_PATH

# Verify Day 5 Model (99.71%)
python -c "import torch; m=torch.load('data/models/efficientnet_b3_final.pth'); print(f'Accuracy: {m[\'val_acc\']:.2f}%')"


## 📁 Repository Structure
```text
deepfake-guard/
├── backend/
│   └── app/
│       └── ml/
│           ├── models/
│           │   ├── meso4.py                    ✅ Day 4
│           │   └── train_baseline.py           ✅ Day 4
│           ├── ensemble/
│           │   ├── efficientnet_model.py       ✅ Day 5
│           │   ├── train_efficientnet.py       ✅ Day 5
│           │   └── train_efficientnet_fast.py  ✅ Day 5
│           ├── adversarial/
│           │   ├── test_fgsm_baseline.py       ✅ Day 8
│           │   ├── test_pgd_baseline.py        ✅ Day 9
│           │   └── train_robust_final.py       ✅ Day 11
│           ├── preprocessing/
│           │   ├── extract_faces_mtcnn.py      ✅ Day 3
│           │   └── generate_test_videos.py     ✅ Day 3
│           └── data/
│               ├── download_celebdf.py         ✅ Day 2
│               └── organize_celebdf.py         ✅ Day 5
├── data/
│   ├── models/
│   │   ├── meso4_baseline_best.pth             ✅ 80% acc
│   │   ├── meso4_pgd_hardened.pth              ✅ 50% robust
│   │   └── efficientnet_b3_final.pth           ✅ 99.71%
│   ├── processed/
│   │   ├── 140k/real_vs_fake/                  ✅ 140k images
│   │   └── celeb-df/faces/                     ✅ 61k faces
│   └── raw/
│       └── celeb-df/videos/                    ✅ Source
├── docs/
│   └── day5.md                                 ✅ Day 5
├── frontend/                                   🔄 Days 22-28
├── docker-compose.yml                          ✅ Day 1
├── requirements.txt                            ✅ Python deps
└── README.md                                   ✅ This file

```   


## 💰 Budget Tracking

| Item                     | Cost          | Status             |
| ------------------------ | ------------- | ------------------ |
| Google Colab Pro (Day 4) | \$12          | ✅ Spent            |
| Google Colab Pro (Day 5) | \$8           | ✅ Spent            |
| **Total Spent**          | **\$20**      | —                  |
| **Remaining**            | **\$130**     | ✅ On track         |
| **Budget**               | **\$150 CAD** | Well within limits |


## 📝 Git Status
Repository: github.com/omi201/deepfake-guard
Branch: main
Last Commit: 77b5dd6 - Day 5 COMPLETE: 99.71% accuracy
Commits: Daily (Days 1-5), no skipped days
Status: Clean working tree, all changes pushed


## 🎯 Next Steps (Day 6)
Objective: Implement Albumentations-based data augmentation
Augmentations: JPEG compression, Gaussian noise, blur, rotation
Target File: backend/app/ml/preprocessing/augmentation.py



