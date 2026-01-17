import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageEnhance
import random
import json
from tqdm import tqdm


def augment_fabric_image(img, augmentation_type):
    """
    Apply augmentation techniques suitable for fabric analysis.
    Preserves fabric characteristics while increasing dataset diversity.
    """
    h, w = img.shape[:2]
    
    if augmentation_type == 'horizontal_flip':
        return cv2.flip(img, 1)
    
    elif augmentation_type == 'vertical_flip':
        return cv2.flip(img, 0)
    
    elif augmentation_type == 'rotate_90':
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    elif augmentation_type == 'rotate_180':
        return cv2.rotate(img, cv2.ROTATE_180)
    
    elif augmentation_type == 'rotate_270':
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    elif augmentation_type == 'brightness_up':
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        enhanced = enhancer.enhance(1.2)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    elif augmentation_type == 'brightness_down':
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        enhanced = enhancer.enhance(0.8)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    elif augmentation_type == 'contrast_up':
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(1.3)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    elif augmentation_type == 'contrast_down':
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(0.7)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    elif augmentation_type == 'gaussian_noise':
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    elif augmentation_type == 'slight_blur':
        return cv2.GaussianBlur(img, (3, 3), 0.5)
    
    elif augmentation_type == 'sharpen':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1.0
        return cv2.filter2D(img, -1, kernel)
    
    elif augmentation_type == 'zoom_in':
        # Zoom in by 10% (crop center and resize back)
        crop_size = int(min(h, w) * 0.9)
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        cropped = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
        return cv2.resize(cropped, (w, h))
    
    elif augmentation_type == 'zoom_out':
        # Zoom out by adding border
        border = int(min(h, w) * 0.05)
        bordered = cv2.copyMakeBorder(img, border, border, border, border, 
                                      cv2.BORDER_REFLECT)
        return cv2.resize(bordered, (w, h))
    
    elif augmentation_type == 'gamma_correction':
        # Gamma correction for lighting variation
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    else:
        return img


def augment_dataset(input_dir="data/preprocessed_dataset", 
                    output_dir="data/augmented_dataset",
                    augmentations_per_image=5,
                    preserve_original=True):
    """
    Augment the dataset with realistic transformations.
    
    Args:
        input_dir: Source directory with images
        output_dir: Destination for augmented dataset
        augmentations_per_image: Number of augmented versions per image
        preserve_original: Keep original images in augmented dataset
    """
    
    project_root = Path(__file__).parent
    input_path = project_root / input_dir
    output_path = project_root / output_dir
    
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original dataset
    csv_path = input_path / "dataset.csv"
    df_original = pd.read_csv(csv_path)
    
    # Augmentation strategies
    augmentation_types = [
        'horizontal_flip', 'vertical_flip',
        'rotate_90', 'rotate_180', 'rotate_270',
        'brightness_up', 'brightness_down',
        'contrast_up', 'contrast_down',
        'gaussian_noise', 'slight_blur', 'sharpen',
        'zoom_in', 'zoom_out', 'gamma_correction'
    ]
    
    print("=" * 100)
    print("📸 FABRIC DATASET AUGMENTATION PIPELINE")
    print("=" * 100)
    print(f"📊 Original images: {len(df_original)}")
    print(f"🔄 Augmentations per image: {augmentations_per_image}")
    print(f"📈 Target dataset size: {len(df_original) * (augmentations_per_image + (1 if preserve_original else 0))}")
    print("=" * 100)
    
    augmented_records = []
    
    # Process each image
    for idx, row in tqdm(df_original.iterrows(), total=len(df_original), desc="Augmenting"):
        img_name = row['image_name']
        img_path = input_path / "images" / img_name
        
        if not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Preserve original
        if preserve_original:
            original_output = images_dir / img_name
            cv2.imwrite(str(original_output), img)
            augmented_records.append({
                'image_name': img_name,
                'gsm': row.get('gsm', np.nan),
                'source': row.get('source', 'unknown'),
                'augmentation': 'original',
                'original_image': img_name
            })
        
        # Generate augmented versions
        selected_augmentations = random.sample(augmentation_types, 
                                              min(augmentations_per_image, len(augmentation_types)))
        
        for aug_idx, aug_type in enumerate(selected_augmentations):
            try:
                augmented_img = augment_fabric_image(img, aug_type)
                
                # Create augmented image name
                base_name = img_name.rsplit('.', 1)[0]
                ext = img_name.rsplit('.', 1)[1]
                aug_name = f"{base_name}_aug_{aug_idx+1}_{aug_type}.{ext}"
                
                aug_output = images_dir / aug_name
                cv2.imwrite(str(aug_output), augmented_img)
                
                augmented_records.append({
                    'image_name': aug_name,
                    'gsm': row.get('gsm', np.nan),
                    'source': row.get('source', 'unknown'),
                    'augmentation': aug_type,
                    'original_image': img_name
                })
                
            except Exception as e:
                print(f"⚠️ Error augmenting {img_name} with {aug_type}: {e}")
                continue
    
    # Create augmented dataset CSV
    df_augmented = pd.DataFrame(augmented_records)
    
    output_csv = output_path / "dataset.csv"
    df_augmented.to_csv(output_csv, index=False)
    
    print(f"\n{'='*100}")
    print(f"✅ AUGMENTATION COMPLETE!")
    print(f"{'='*100}")
    print(f"📊 Original images: {len(df_original)}")
    print(f"📈 Augmented dataset: {len(df_augmented)} images")
    print(f"🔄 Increase factor: {len(df_augmented) / len(df_original):.1f}x")
    print(f"📁 Output: {output_path}")
    print(f"{'='*100}")
    
    # Save augmentation summary
    augmentation_summary = {
        'original_count': len(df_original),
        'augmented_count': len(df_augmented),
        'increase_factor': len(df_augmented) / len(df_original),
        'augmentations_per_image': augmentations_per_image,
        'preserve_original': preserve_original,
        'augmentation_types_used': augmentation_types,
        'augmentation_distribution': df_augmented['augmentation'].value_counts().to_dict()
    }
    
    summary_path = output_path / "augmentation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(augmentation_summary, f, indent=2)
    
    # Statistics by source
    print("\n📊 Dataset Distribution:")
    print(df_augmented.groupby('source')['image_name'].count())
    
    # GSM distribution
    if 'gsm' in df_augmented.columns:
        print("\n📊 GSM Statistics:")
        print(f"Mean GSM: {df_augmented['gsm'].mean():.2f}")
        print(f"Std GSM: {df_augmented['gsm'].std():.2f}")
        print(f"Min GSM: {df_augmented['gsm'].min():.2f}")
        print(f"Max GSM: {df_augmented['gsm'].max():.2f}")
    
    return df_augmented


def create_stratified_splits(dataset_dir="data/augmented_dataset", 
                            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create stratified train/val/test splits ensuring augmented versions 
    of the same original image stay in the same split.
    """
    
    project_root = Path(__file__).parent
    dataset_path = project_root / dataset_dir
    
    csv_path = dataset_path / "dataset.csv"
    df = pd.read_csv(csv_path)
    
    # Group by original image
    unique_originals = df['original_image'].unique()
    np.random.shuffle(unique_originals)
    
    n_train = int(len(unique_originals) * train_ratio)
    n_val = int(len(unique_originals) * val_ratio)
    
    train_originals = unique_originals[:n_train]
    val_originals = unique_originals[n_train:n_train+n_val]
    test_originals = unique_originals[n_train+n_val:]
    
    df['split'] = 'test'
    df.loc[df['original_image'].isin(train_originals), 'split'] = 'train'
    df.loc[df['original_image'].isin(val_originals), 'split'] = 'val'
    
    # Save updated CSV with splits
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*100)
    print("📂 DATASET SPLITS CREATED")
    print("="*100)
    print(f"🟢 Train: {len(df[df['split']=='train'])} images ({len(train_originals)} unique originals)")
    print(f"🟡 Val: {len(df[df['split']=='val'])} images ({len(val_originals)} unique originals)")
    print(f"🔴 Test: {len(df[df['split']=='test'])} images ({len(test_originals)} unique originals)")
    print("="*100)
    
    # Save individual split CSVs
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_csv = dataset_path / f"dataset_{split}.csv"
        split_df.to_csv(split_csv, index=False)
        print(f"✅ Saved: {split_csv}")
    
    return df


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Augment dataset
    df_augmented = augment_dataset(
        input_dir="data/preprocessed_dataset",
        output_dir="data/augmented_dataset",
        augmentations_per_image=5,  # Generate 5 augmented versions per image
        preserve_original=True       # Keep original images
    )
    
    # Create train/val/test splits
    df_splits = create_stratified_splits(
        dataset_dir="data/augmented_dataset",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print("\n✅ COMPLETE! Ready for training.")
