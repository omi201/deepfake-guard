#!/usr/bin/env python3
from pathlib import Path
import sys

def verify():
    base = Path("data/processed/celeb-df/faces")
    if not base.exists():
        print("❌ FAIL: Directory does not exist")
        return 1
    
    real = len(list((base / "real").glob("*.jpg"))) if (base / "real").exists() else 0
    fake = len(list((base / "fake").glob("*.jpg"))) if (base / "fake").exists() else 0
    total = real + fake
    
    print(f"Real faces: {real}")
    print(f"Fake faces: {fake}")
    print(f"Total: {total}")
    
    if total > 0:
        print(f"\n✅ PASS: {total} faces extracted (Pipeline functional)")
        return 0
    else:
        print(f"\n❌ FAIL: No faces extracted")
        return 1

if __name__ == "__main__":
    sys.exit(verify())
