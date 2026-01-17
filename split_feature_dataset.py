"""
Split the feature_extracted_dataset into train/validation/test sets.
This avoids augmentation and uses only the original 177 images with extracted features.
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path("data/feature_extracted_dataset")
OUTPUT_DIR = Path("data/split_feature_dataset")

IMAGES_DIR = BASE_DIR / "images"
CSV_FILE = BASE_DIR / "dataset_with_features.csv"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "test" / "images").mkdir(parents=True, exist_ok=True)

print("="*80)
print("📊 SPLITTING FEATURE-EXTRACTED DATASET")
print("="*80)

# Load dataset
df = pd.read_csv(CSV_FILE)
print(f"\nTotal samples: {len(df)}")
print(f"Features: {df.shape[1] - 3} (excluding image_name, gsm, source)")

# Check for missing values in features
feature_cols = [col for col in df.columns if col not in ['image_name', 'gsm', 'source']]
missing_counts = df[feature_cols].isna().sum()
if missing_counts.sum() > 0:
    print(f"\n⚠️ Found {missing_counts.sum()} missing values in features")
    print("Filling with median values...")
    for col in feature_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    print("✅ Missing values filled")

# GSM distribution
print(f"\n📈 GSM Distribution:")
print(f"  Mean: {df['gsm'].mean():.2f}")
print(f"  Std:  {df['gsm'].std():.2f}")
print(f"  Min:  {df['gsm'].min():.0f}")
print(f"  Max:  {df['gsm'].max():.0f}")

# Split strategy: 70% train, 15% val, 15% test
# Using stratified split to maintain GSM distribution
print(f"\n🔀 Splitting dataset...")
print(f"  Train: 70%")
print(f"  Val:   15%")
print(f"  Test:  15%")

# Create GSM bins for stratification
df['gsm_bin'] = pd.cut(df['gsm'], bins=10, labels=False)

# First split: train vs (val + test)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    random_state=RANDOM_SEED,
    stratify=df['gsm_bin']
)

# Second split: val vs test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,  # 50% of 30% = 15% of total
    random_state=RANDOM_SEED,
    stratify=temp_df['gsm_bin']
)

# Remove temporary column
train_df = train_df.drop('gsm_bin', axis=1)
val_df = val_df.drop('gsm_bin', axis=1)
test_df = test_df.drop('gsm_bin', axis=1)

print(f"\n✅ Split complete:")
print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Verify GSM distribution in each split
print(f"\n📊 GSM Distribution per split:")
print(f"  Train - Mean: {train_df['gsm'].mean():.2f}, Std: {train_df['gsm'].std():.2f}, Range: [{train_df['gsm'].min():.0f}, {train_df['gsm'].max():.0f}]")
print(f"  Val   - Mean: {val_df['gsm'].mean():.2f}, Std: {val_df['gsm'].std():.2f}, Range: [{val_df['gsm'].min():.0f}, {val_df['gsm'].max():.0f}]")
print(f"  Test  - Mean: {test_df['gsm'].mean():.2f}, Std: {test_df['gsm'].std():.2f}, Range: [{test_df['gsm'].min():.0f}, {test_df['gsm'].max():.0f}]")

# Copy images to respective folders
print(f"\n📁 Copying images...")

def copy_images(dataframe, split_name):
    """Copy images for a specific split."""
    dest_dir = OUTPUT_DIR / split_name / "images"
    copied = 0
    missing = 0
    
    for img_name in dataframe['image_name']:
        src_path = IMAGES_DIR / img_name
        dest_path = dest_dir / img_name
        
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied += 1
        else:
            print(f"  ⚠️ Missing: {img_name}")
            missing += 1
    
    print(f"  {split_name.capitalize()}: {copied} images copied, {missing} missing")
    return copied, missing

train_copied, train_missing = copy_images(train_df, 'train')
val_copied, val_missing = copy_images(val_df, 'val')
test_copied, test_missing = copy_images(test_df, 'test')

# Save CSV files
print(f"\n💾 Saving CSV files...")
train_df.to_csv(OUTPUT_DIR / "train" / "dataset_train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val" / "dataset_val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test" / "dataset_test.csv", index=False)

# Also save combined CSV for reference
df.to_csv(OUTPUT_DIR / "dataset_all.csv", index=False)

print(f"  ✅ train/dataset_train.csv ({len(train_df)} rows)")
print(f"  ✅ val/dataset_val.csv ({len(val_df)} rows)")
print(f"  ✅ test/dataset_test.csv ({len(test_df)} rows)")
print(f"  ✅ dataset_all.csv ({len(df)} rows)")

# Create summary report
summary = {
    'total_samples': len(df),
    'features_count': len(feature_cols),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'train_percentage': len(train_df) / len(df) * 100,
    'val_percentage': len(val_df) / len(df) * 100,
    'test_percentage': len(test_df) / len(df) * 100,
    'gsm_range': [float(df['gsm'].min()), float(df['gsm'].max())],
    'train_gsm_mean': float(train_df['gsm'].mean()),
    'val_gsm_mean': float(val_df['gsm'].mean()),
    'test_gsm_mean': float(test_df['gsm'].mean()),
    'random_seed': RANDOM_SEED
}

import json
with open(OUTPUT_DIR / "split_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  ✅ split_summary.json")

print("\n" + "="*80)
print("✅ DATASET SPLIT COMPLETE")
print("="*80)
print(f"\n📂 Output location: {OUTPUT_DIR.absolute()}")
print(f"\n💡 Next steps:")
print(f"  1. Update notebook to use: {OUTPUT_DIR.absolute()}")
print(f"  2. Set IMAGES_PATH to train/val/test subdirectories")
print(f"  3. Load CSV files from train/val/test folders")
print(f"  4. Train model with original {len(df)} samples (no augmentation)")
print("="*80)
