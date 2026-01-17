import os
import pandas as pd
import shutil
from pathlib import Path

def create_combined_dataset():
    """
    Create a robust dataset combining FabricNet images (130) and manually collected GSM images (50).
    """
    
    # Paths
    project_root = Path(__file__).parent
    fabricnet_excel = project_root / "data" / "FabricNet_parameters.xlsx"
    gsm_dir = project_root / "data" / "gsm" / "gsm"
    
    # Output paths
    output_dir = project_root / "data" / "combined_dataset"
    output_images_dir = output_dir / "images"
    output_csv = output_dir / "dataset.csv"
    
    # Create output directory
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🔍 Step 1: Loading FabricNet data...")
    print("=" * 80)
    
    # Load FabricNet data
    df_fabricnet = pd.read_excel(fabricnet_excel)
    print(f"✅ Loaded FabricNet data: {len(df_fabricnet)} rows")
    print(f"📊 Columns: {df_fabricnet.columns.tolist()}")
    
    # Check for specific_mass or similar columns
    gsm_column = None
    for col in df_fabricnet.columns:
        if 'specific' in col.lower() and 'mass' in col.lower():
            gsm_column = col
            break
    
    if gsm_column:
        print(f"✅ Found GSM column: '{gsm_column}'")
    else:
        print("⚠️  No 'specific mass' column found. Available columns:")
        print(df_fabricnet.columns.tolist())
        # Try to find any GSM-related column
        for col in df_fabricnet.columns:
            if 'gsm' in col.lower() or 'gram' in col.lower() or 'weight' in col.lower():
                gsm_column = col
                print(f"ℹ️  Using column: '{gsm_column}' as GSM")
                break
    
    print("\n" + "=" * 80)
    print("🔍 Step 2: Scanning manually collected GSM images...")
    print("=" * 80)
    
    # Scan GSM directory
    gsm_images = []
    if gsm_dir.exists():
        for subfolder in gsm_dir.iterdir():
            if subfolder.is_dir():
                # Extract GSM value from folder name (e.g., "1-173" -> 173)
                folder_name = subfolder.name
                if '-' in folder_name:
                    gsm_value = folder_name.split('-')[1]
                    
                    # Count images in this folder
                    image_files = list(subfolder.glob('*.jpg')) + list(subfolder.glob('*.png')) + \
                                  list(subfolder.glob('*.jpeg')) + list(subfolder.glob('*.bmp'))
                    
                    for img_file in image_files:
                        gsm_images.append({
                            'original_path': img_file,
                            'gsm_value': gsm_value,
                            'folder': folder_name
                        })
    
    print(f"✅ Found {len(gsm_images)} manually collected images across {len(list(gsm_dir.iterdir()))} folders")
    
    print("\n" + "=" * 80)
    print("🔧 Step 3: Creating combined dataset...")
    print("=" * 80)
    
    # Create combined dataset
    dataset_records = []
    image_counter = 1
    
    # Process FabricNet images
    print(f"\n📦 Processing FabricNet images...")
    fabricnet_image_dir = project_root / "data"
    
    # FabricNet images are named W001.jpg to W130.jpg
    for idx, row in df_fabricnet.iterrows():
        # Image names follow pattern W001.jpg, W002.jpg, etc.
        img_name = f"W{idx + 1:03d}.jpg"
        img_path = fabricnet_image_dir / img_name
        
        if img_path.exists():
            # Copy and rename image
            new_name = f"fabricnet_{image_counter:04d}.jpg"
            new_path = output_images_dir / new_name
            shutil.copy2(img_path, new_path)
            
            # Get GSM value
            gsm_val = row[gsm_column] if gsm_column and gsm_column in row else None
            
            dataset_records.append({
                'image_name': new_name,
                'gsm': gsm_val,
                'source': 'fabricnet',
                'original_name': img_path.name,
                'data_index': idx
            })
            
            image_counter += 1
    
    print(f"✅ Processed {len([r for r in dataset_records if r['source'] == 'fabricnet'])} FabricNet images")
    
    # Process manually collected GSM images
    print(f"\n📦 Processing manually collected GSM images...")
    for img_info in gsm_images:
        img_path = img_info['original_path']
        ext = img_path.suffix
        
        # Copy and rename image
        new_name = f"gsm_manual_{image_counter:04d}{ext}"
        new_path = output_images_dir / new_name
        shutil.copy2(img_path, new_path)
        
        dataset_records.append({
            'image_name': new_name,
            'gsm': img_info['gsm_value'],
            'source': 'manual_collection',
            'original_name': img_path.name,
            'folder': img_info['folder']
        })
        
        image_counter += 1
    
    print(f"✅ Processed {len([r for r in dataset_records if r['source'] == 'manual_collection'])} manually collected images")
    
    # Create DataFrame and save to CSV
    df_combined = pd.DataFrame(dataset_records)
    
    # Convert GSM to numeric for consistency
    df_combined['gsm'] = pd.to_numeric(df_combined['gsm'], errors='coerce')
    
    df_combined.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 80)
    print("✅ Dataset Creation Complete!")
    print("=" * 80)
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️  Images saved to: {output_images_dir}")
    print(f"📊 CSV saved to: {output_csv}")
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total images: {len(dataset_records)}")
    print(f"   - FabricNet images: {len([r for r in dataset_records if r['source'] == 'fabricnet'])}")
    print(f"   - Manual images: {len([r for r in dataset_records if r['source'] == 'manual_collection'])}")
    print(f"\n📋 CSV Preview:")
    print(df_combined.head(10))
    print(f"\n📊 GSM value distribution:")
    print(df_combined['gsm'].value_counts().sort_index())
    print(f"\n📊 GSM Statistics:")
    print(df_combined['gsm'].describe())

if __name__ == "__main__":
    create_combined_dataset()
