import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json

def preprocess_images(
    input_dir="data/combined_dataset",
    output_dir="data/preprocessed_dataset",
    target_size=(512, 512),
    maintain_aspect=True,
    apply_padding=True,
    save_stats=True
):
    """
    Preprocess images for training:
    - Resize to consistent dimensions
    - Maintain aspect ratio with padding
    - Normalize pixel values
    - Save preprocessing statistics
    """
    
    project_root = Path(__file__).parent
    input_path = project_root / input_dir
    output_path = project_root / output_dir
    
    # Create output directories
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset CSV
    csv_path = input_path / "dataset.csv"
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("🖼️  IMAGE PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"📊 Total images to process: {len(df)}")
    print(f"🎯 Target size: {target_size}")
    print(f"📐 Maintain aspect ratio: {maintain_aspect}")
    print(f"🔲 Apply padding: {apply_padding}")
    print("=" * 80)
    
    # Statistics tracking
    stats = {
        'original_sizes': [],
        'processed_count': 0,
        'failed_count': 0,
        'target_size': target_size,
        'maintain_aspect': maintain_aspect,
        'apply_padding': apply_padding
    }
    
    # Process each image
    processed_records = []
    
    for idx, row in df.iterrows():
        try:
            # Load original image
            img_name = row['image_name']
            img_path = input_path / "images" / img_name
            
            if not img_path.exists():
                print(f"⚠️  Image not found: {img_name}")
                stats['failed_count'] += 1
                continue
            
            img = Image.open(img_path)
            original_size = img.size
            stats['original_sizes'].append(original_size)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize with aspect ratio preservation
            if maintain_aspect:
                img_resized = resize_with_padding(img, target_size, apply_padding)
            else:
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save preprocessed image
            output_img_path = images_dir / img_name
            img_resized.save(output_img_path, 'JPEG', quality=95)
            
            # Record metadata
            processed_records.append({
                'image_name': img_name,
                'gsm': row['gsm'],
                'source': row['source'],
                'original_name': row['original_name'],
                'original_width': original_size[0],
                'original_height': original_size[1],
                'processed_width': target_size[0],
                'processed_height': target_size[1],
                'aspect_ratio': original_size[0] / original_size[1],
                'data_index': row.get('data_index', ''),
                'folder': row.get('folder', '')
            })
            
            stats['processed_count'] += 1
            
            if (idx + 1) % 20 == 0:
                print(f"✅ Processed {idx + 1}/{len(df)} images...")
                
        except Exception as e:
            print(f"❌ Error processing {row['image_name']}: {e}")
            stats['failed_count'] += 1
    
    # Create preprocessed dataset CSV
    df_processed = pd.DataFrame(processed_records)
    output_csv = output_path / "dataset.csv"
    df_processed.to_csv(output_csv, index=False)
    
    # Calculate and save statistics
    if stats['original_sizes']:
        widths = [s[0] for s in stats['original_sizes']]
        heights = [s[1] for s in stats['original_sizes']]
        
        stats['original_size_stats'] = {
            'min_width': min(widths),
            'max_width': max(widths),
            'mean_width': np.mean(widths),
            'min_height': min(heights),
            'max_height': max(heights),
            'mean_height': np.mean(heights)
        }
        
        # Remove the list to make JSON serializable
        stats['original_sizes'] = f"{len(stats['original_sizes'])} unique sizes recorded"
    
    if save_stats:
        stats_path = output_path / "preprocessing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"✅ Successfully processed: {stats['processed_count']} images")
    print(f"❌ Failed: {stats['failed_count']} images")
    print(f"📁 Output directory: {output_path}")
    print(f"🖼️  Images: {images_dir}")
    print(f"📊 Dataset CSV: {output_csv}")
    
    if save_stats:
        print(f"📈 Statistics: {stats_path}")
    
    print("\n📊 Original Image Size Statistics:")
    if 'original_size_stats' in stats:
        size_stats = stats['original_size_stats']
        print(f"   Width:  {size_stats['min_width']}-{size_stats['max_width']} "
              f"(avg: {size_stats['mean_width']:.1f})")
        print(f"   Height: {size_stats['min_height']}-{size_stats['max_height']} "
              f"(avg: {size_stats['mean_height']:.1f})")
    
    print("\n📊 GSM Distribution:")
    print(df_processed['gsm'].value_counts().sort_index())
    
    print("\n📊 Dataset Summary:")
    print(df_processed.groupby('source').size())
    
    return df_processed, stats


def resize_with_padding(img, target_size, apply_padding=True):
    """
    Resize image maintaining aspect ratio, optionally with padding.
    """
    target_w, target_h = target_size
    original_w, original_h = img.size
    
    # Calculate scaling factor to fit within target size
    scale = min(target_w / original_w, target_h / original_h)
    
    # Calculate new size maintaining aspect ratio
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # Resize image
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    if apply_padding:
        # Create a new image with target size and paste resized image in center
        new_img = Image.new('RGB', target_size, (255, 255, 255))  # White padding
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        return new_img
    else:
        return img_resized


def analyze_images(dataset_dir="data/combined_dataset"):
    """
    Analyze original images to determine optimal preprocessing parameters.
    """
    project_root = Path(__file__).parent
    dataset_path = project_root / dataset_dir
    images_path = dataset_path / "images"
    
    print("=" * 80)
    print("🔍 ANALYZING IMAGES")
    print("=" * 80)
    
    sizes = []
    aspect_ratios = []
    
    for img_file in images_path.glob("*.jpg"):
        try:
            img = Image.open(img_file)
            w, h = img.size
            sizes.append((w, h))
            aspect_ratios.append(w / h)
        except Exception as e:
            print(f"Error reading {img_file.name}: {e}")
    
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        print(f"📊 Total images analyzed: {len(sizes)}")
        print(f"\n📏 Width Statistics:")
        print(f"   Min: {min(widths)}px")
        print(f"   Max: {max(widths)}px")
        print(f"   Mean: {np.mean(widths):.1f}px")
        print(f"   Median: {np.median(widths):.1f}px")
        
        print(f"\n📏 Height Statistics:")
        print(f"   Min: {min(heights)}px")
        print(f"   Max: {max(heights)}px")
        print(f"   Mean: {np.mean(heights):.1f}px")
        print(f"   Median: {np.median(heights):.1f}px")
        
        print(f"\n📐 Aspect Ratio Statistics:")
        print(f"   Min: {min(aspect_ratios):.2f}")
        print(f"   Max: {max(aspect_ratios):.2f}")
        print(f"   Mean: {np.mean(aspect_ratios):.2f}")
        
        print(f"\n💡 Recommended Target Sizes:")
        print(f"   Standard: 512x512 (balanced)")
        print(f"   High Res: 1024x1024 (better quality, slower)")
        print(f"   Low Res: 256x256 (faster, less detail)")
    
    print("=" * 80)
    return sizes, aspect_ratios


if __name__ == "__main__":
    # First, analyze images to understand the data
    print("Step 1: Analyzing original images...\n")
    analyze_images()
    
    print("\n\nStep 2: Preprocessing images...\n")
    
    # Preprocess with recommended settings
    df_processed, stats = preprocess_images(
        input_dir="data/combined_dataset",
        output_dir="data/preprocessed_dataset",
        target_size=(512, 512),  # Standard size for deep learning
        maintain_aspect=True,     # Preserve aspect ratio
        apply_padding=True,       # Add white padding to reach target size
        save_stats=True
    )
    
    print("\n✅ All done! Your dataset is ready for training.")
