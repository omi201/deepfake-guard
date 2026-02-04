# Adversarial Deepfake Detection System (ADDS)

Security Objective
Build a hardened deepfake detection system resistant to adversarial evasion attacks.

## Structure
- `backend/`: Python FastAPI application with ML models
- `frontend/`: React dashboard for forensic visualization
- `data/`: Datasets (raw = immutable, processed = derived)
- `docs/`: Security analysis reports and architecture diagrams

## Current Status
- ✅ **Day 1**: Docker infrastructure (Postgres + Redis) - COMPLETE
- ✅ **Day 2**: Dataset verification (140k images) - COMPLETE
- 🔄 **Day 3**: Data loading pipeline - PENDING

## Datasets Verified
- **140k Real vs Fake Faces**: 140,000 images (50k train, 10k test, 10k valid per class)
- **Celeb-DF**: Infrastructure ready (optional download pending)

## Models
- Meso4 Baseline: 80% accuracy ✅
- Meso4 PGD-Hardened: 50% robust accuracy ✅
