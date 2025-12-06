#!/usr/bin/env python3
"""
Quick script to check how many classes a saved model expects
"""
import torch
from pathlib import Path

# Path to your model
model_path = input("Enter path to model.pth: ")

if not Path(model_path).exists():
    print(f"❌ File not found: {model_path}")
    exit(1)

# Load state dict
state_dict = torch.load(model_path, map_location='cpu')

# Find classifier layers (different architectures have different names)
classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k or 'head' in k]

print(f"\n📊 Model Analysis: {model_path}")
print("=" * 80)

for key in classifier_keys:
    tensor = state_dict[key]
    print(f"\n{key}:")
    print(f"  Shape: {tensor.shape}")

    # For weights (2D tensor), first dimension is output classes
    if len(tensor.shape) == 2:
        out_features = tensor.shape[0]
        print(f"  ✅ Output classes: {out_features}")
    # For bias (1D tensor), length is output classes
    elif len(tensor.shape) == 1:
        out_features = tensor.shape[0]
        print(f"  ✅ Output classes: {out_features}")

print("\n" + "=" * 80)
