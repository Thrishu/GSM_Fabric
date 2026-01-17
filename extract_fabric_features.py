import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift, fft as fft1d, fftfreq
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
import time
warnings.filterwarnings('ignore')


def process_single_image(args):
    """Process a single image - designed for parallel execution."""
    idx, row, dataset_path, save_visualizations, viz_dir = args
    
    img_name = row['image_name']
    features = {}
    
    try:
        img_path = dataset_path / "images" / img_name
        
        if not img_path.exists():
            return {'image_name': img_name, 'gsm': row.get('gsm', np.nan), 'source': row.get('source', 'unknown')}
        
        img = cv2.imread(str(img_path))
        if img is None:
            return {'image_name': img_name, 'gsm': row.get('gsm', np.nan), 'source': row.get('source', 'unknown')}
            
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        try:
            weft_features = extract_thread_count(img_gray, direction='horizontal')
            warp_features = extract_thread_count(img_gray, direction='vertical')
            features.update(weft_features)
            features.update(warp_features)
        except: pass
        
        try:
            features.update(extract_yarn_density(img_gray))
        except: pass
        
        try:
            features.update(extract_thread_spacing(img_gray))
        except: pass
        
        try:
            features.update(extract_texture_features(img_gray))
        except: pass
        
        try:
            features.update(extract_frequency_features(img_gray))
        except: pass
        
        try:
            features.update(extract_structure_features(img_gray))
        except: pass
        
        try:
            features.update(extract_edge_features(img_gray))
        except: pass
        
        try:
            features.update(extract_color_features(img))
        except: pass
        
        if save_visualizations:
            try:
                viz_img = create_feature_visualization(img, img_gray, features, img_name)
                cv2.imwrite(str(viz_dir / f"viz_{img_name}"), viz_img)
            except: pass
        
        return {'image_name': img_name, 'gsm': row.get('gsm', np.nan), 'source': row.get('source', 'unknown'), **features}
        
    except Exception as e:
        return {'image_name': img_name, 'gsm': row.get('gsm', np.nan), 'source': row.get('source', 'unknown')}


def extract_all_fabric_features(dataset_dir="data/preprocessed_dataset", output_dir="data/feature_extracted_dataset", 
                                 save_visualizations=True, n_workers=None):
    """Extract comprehensive fabric features using parallel processing."""
    
    project_root = Path(__file__).parent
    dataset_path = project_root / dataset_dir
    output_path = project_root / output_dir
    
    images_dir = output_path / "images"
    viz_dir = output_path / "visualizations"
    images_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = dataset_path / "dataset.csv"
    df = pd.read_csv(csv_path)
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print("=" * 100)
    print("🔬 FABRIC FEATURE EXTRACTION PIPELINE (PARALLEL MODE)")
    print("=" * 100)
    print(f"📊 Total images: {len(df)}")
    print(f"⚡ Workers: {n_workers}")
    print(f"📸 Visualizations: {'Enabled' if save_visualizations else 'Disabled'}")
    print("=" * 100)
    
    args_list = [(idx, row, dataset_path, save_visualizations, viz_dir) for idx, row in df.iterrows()]
    
    feature_records = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(process_single_image, args): args[0] for args in args_list}
        
        for future in as_completed(future_to_idx):
            try:
                record = future.result()
                feature_records.append(record)
                completed += 1
                
                if completed % 20 == 0 or completed == len(df):
                    progress = (completed / len(df)) * 100
                    print(f"⚡ Progress: {completed}/{len(df)} ({progress:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print(f"\n{'='*100}")
    print(f"✅ Completed: {completed} images")
    print(f"{'='*100}\n")
    
    df_features = pd.DataFrame(feature_records)
    
    print(f"📊 DataFrame: {len(df_features)} rows, {len(df_features.columns)} columns")
    
    output_csv = output_path / "dataset_with_features.csv"
    df_features.to_csv(output_csv, index=False)
    
    print(f"✅ CSV saved: {output_csv}")
    
    df_verify = pd.read_csv(output_csv)
    print(f"✅ Verified: {len(df_verify)} rows, {len(df_verify.columns)} columns")
    
    feature_summary = {
        'total_images': len(df_features),
        'total_features': len(df_features.columns) - 3,
        'feature_names': [col for col in df_features.columns if col not in ['image_name', 'gsm', 'source']],
    }
    
    if len(df_features.columns) > 3:
        try:
            feature_summary['feature_statistics'] = df_features.describe().to_dict()
        except:
            pass
    
    summary_path = output_path / "feature_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(feature_summary, f, indent=2)
    
    print(f"\n✅ COMPLETE!")
    print(f"📊 Features: {len(df_features.columns) - 3}")
    print(f"📁 Output: {output_path}")
    
    return df_features


def extract_thread_count(img_gray, direction='horizontal'):
    """Extract thread count using multiple methods."""
    features = {}
    prefix = 'weft_' if direction == 'horizontal' else 'warp_'
    
    h, w = img_gray.shape
    
    thread_counts_autocorr = []
    spacings = []
    
    roi_positions = [(h//4, 3*h//4, w//4, 3*w//4), (h//6, 5*h//6, w//6, 5*w//6), (h//3, 2*h//3, w//3, 2*w//3)]
    
    for y1, y2, x1, x2 in roi_positions:
        roi = img_gray[y1:y2, x1:x2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        roi_enhanced = clahe.apply(roi)
        roi_filtered = cv2.bilateralFilter(roi_enhanced, 9, 75, 75)
        
        profile = np.mean(roi_filtered, axis=1) if direction == 'horizontal' else np.mean(roi_filtered, axis=0)
        profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile) + 1e-8)
        profile = signal.savgol_filter(profile, window_length=min(11, len(profile)//2*2-1), polyorder=3)
        
        autocorr = signal.correlate(profile, profile, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / np.max(autocorr)
        
        min_distance = max(2, len(profile) // 100)
        peaks, _ = signal.find_peaks(autocorr, distance=min_distance, prominence=0.05, height=0.3)
        
        if len(peaks) > 2:
            valid_peaks = peaks[1:]
            if len(valid_peaks) > 1:
                spacing = np.median(np.diff(valid_peaks[:min(15, len(valid_peaks))]))
                if spacing > 0:
                    thread_count = len(profile) / spacing
                    thread_counts_autocorr.append(thread_count)
                    spacings.append(spacing)
    
    thread_counts_fft = []
    roi = img_gray[h//4:3*h//4, w//4:3*w//4]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_enhanced = clahe.apply(roi)
    
    profile = np.mean(roi_enhanced, axis=1) if direction == 'horizontal' else np.mean(roi_enhanced, axis=0)
    profile = signal.detrend(profile)
    profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile) + 1e-8)
    
    window = signal.windows.hann(len(profile))
    profile_windowed = profile * window
    
    fft_result = np.abs(fft1d(profile_windowed))
    freqs = fftfreq(len(profile_windowed))
    
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_result[:len(fft_result)//2]
    
    fft_peaks, _ = signal.find_peaks(positive_fft, prominence=np.max(positive_fft)*0.1)
    
    if len(fft_peaks) > 0:
        fft_peaks = fft_peaks[fft_peaks > 0]
        if len(fft_peaks) > 0:
            dominant_peak_idx = fft_peaks[np.argmax(positive_fft[fft_peaks])]
            dominant_freq = positive_freqs[dominant_peak_idx]
            if dominant_freq > 0:
                thread_count_fft = abs(dominant_freq * len(profile))
                thread_counts_fft.append(thread_count_fft)
    
    roi = img_gray[h//4:3*h//4, w//4:3*w//4]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    roi_enhanced = clahe.apply(roi)
    
    if direction == 'horizontal':
        kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    else:
        kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    
    edges = cv2.filter2D(roi_enhanced, -1, kernel)
    edges = np.abs(edges)
    
    edge_profile = np.mean(edges, axis=1) if direction == 'horizontal' else np.mean(edges, axis=0)
    edge_profile = signal.savgol_filter(edge_profile, window_length=min(11, len(edge_profile)//2*2-1), polyorder=3)
    
    peaks, _ = signal.find_peaks(edge_profile, distance=max(2, len(edge_profile)//100), prominence=np.max(edge_profile)*0.1)
    direct_count = len(peaks)
    
    all_counts = []
    
    if thread_counts_autocorr:
        autocorr_count = np.median(thread_counts_autocorr)
        all_counts.append(autocorr_count)
        features[f'{prefix}count_autocorr'] = autocorr_count
    else:
        features[f'{prefix}count_autocorr'] = 0
    
    if thread_counts_fft:
        fft_count = np.median(thread_counts_fft)
        all_counts.append(fft_count)
        features[f'{prefix}count_fft'] = fft_count
    else:
        features[f'{prefix}count_fft'] = 0
    
    if direct_count > 0:
        all_counts.append(direct_count)
        features[f'{prefix}count_direct'] = direct_count
    else:
        features[f'{prefix}count_direct'] = 0
    
    if all_counts:
        final_count = np.median(all_counts)
        if final_count < 5:
            final_count = max(all_counts) if all_counts else 0
        elif final_count > 300:
            final_count = min(all_counts) if all_counts else 0
    else:
        final_count = 0
    
    features[f'{prefix}count'] = final_count
    
    if spacings:
        features[f'{prefix}spacing_avg'] = np.median(spacings)
        features[f'{prefix}spacing_std'] = np.std(spacings)
        features[f'{prefix}spacing_cv'] = np.std(spacings) / (np.mean(spacings) + 1e-8)
    else:
        features[f'{prefix}spacing_avg'] = 0
        features[f'{prefix}spacing_std'] = 0
        features[f'{prefix}spacing_cv'] = 0
    
    if len(all_counts) > 1:
        count_std = np.std(all_counts)
        count_mean = np.mean(all_counts)
        confidence = 1.0 / (1.0 + count_std / (count_mean + 1e-8))
        features[f'{prefix}confidence'] = confidence
    else:
        features[f'{prefix}confidence'] = 0.5
    
    return features


def extract_yarn_density(img_gray):
    features = {}
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features['yarn_edge_density'] = edge_density
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    features['yarn_component_count'] = num_labels - 1
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        features['yarn_avg_area'] = np.mean(areas)
        features['yarn_std_area'] = np.std(areas)
        features['yarn_density_uniformity'] = 1.0 / (1.0 + np.std(areas) / (np.mean(areas) + 1e-8))
    else:
        features['yarn_avg_area'] = 0
        features['yarn_std_area'] = 0
        features['yarn_density_uniformity'] = 0
    
    return features


def extract_thread_spacing(img_gray):
    features = {}
    edges = cv2.Canny(img_gray, 50, 150)
    
    lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines_h is not None:
        y_positions = sorted([line[0][1] for line in lines_h])
        h_spacing = np.diff(y_positions) if len(y_positions) > 1 else [0]
        features['weft_spacing_uniformity'] = 1.0 / (1.0 + np.std(h_spacing) / (np.mean(h_spacing) + 1e-8))
        features['weft_lines_detected'] = len(lines_h)
    else:
        features['weft_spacing_uniformity'] = 0
        features['weft_lines_detected'] = 0
    
    lines_v = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=50, minLineLength=30, maxLineGap=10)
    if lines_v is not None:
        x_positions = sorted([line[0][0] for line in lines_v])
        v_spacing = np.diff(x_positions) if len(x_positions) > 1 else [0]
        features['warp_spacing_uniformity'] = 1.0 / (1.0 + np.std(v_spacing) / (np.mean(v_spacing) + 1e-8))
        features['warp_lines_detected'] = len(lines_v)
    else:
        features['warp_spacing_uniformity'] = 0
        features['warp_lines_detected'] = 0
    
    return features


def extract_texture_features(img_gray):
    features = {}
    dx = ndimage.sobel(img_gray, axis=1)
    dy = ndimage.sobel(img_gray, axis=0)
    features['texture_contrast'] = np.sqrt(dx**2 + dy**2).mean()
    
    local_std = ndimage.generic_filter(img_gray, np.std, size=5)
    features['texture_homogeneity'] = 1.0 / (1.0 + local_std.mean())
    
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    features['texture_energy'] = np.sum(hist**2)
    
    hist = hist[hist > 0]
    features['texture_entropy'] = -np.sum(hist * np.log2(hist))
    
    lbp = compute_lbp(img_gray)
    features['texture_lbp_mean'] = lbp.mean()
    features['texture_lbp_std'] = lbp.std()
    
    return features


def compute_lbp(img_gray, radius=1):
    h, w = img_gray.shape
    lbp = np.zeros_like(img_gray)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = img_gray[i, j]
            code = 0
            code |= (img_gray[i-1, j-1] >= center) << 7
            code |= (img_gray[i-1, j] >= center) << 6
            code |= (img_gray[i-1, j+1] >= center) << 5
            code |= (img_gray[i, j+1] >= center) << 4
            code |= (img_gray[i+1, j+1] >= center) << 3
            code |= (img_gray[i+1, j] >= center) << 2
            code |= (img_gray[i+1, j-1] >= center) << 1
            code |= (img_gray[i, j-1] >= center) << 0
            lbp[i, j] = code
    
    return lbp


def extract_frequency_features(img_gray):
    features = {}
    fft = fft2(img_gray)
    fft_shifted = fftshift(fft)
    magnitude = np.abs(fft_shifted)
    magnitude_log = np.log1p(magnitude)
    
    h, w = magnitude_log.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    radial_profile = ndimage.mean(magnitude_log, labels=r, index=np.arange(0, r.max() + 1))
    
    features['freq_dc_component'] = magnitude_log[center[0], center[1]]
    features['freq_high_freq_energy'] = np.sum(magnitude_log[0:h//4, :]) + np.sum(magnitude_log[3*h//4:, :])
    features['freq_low_freq_energy'] = np.sum(magnitude_log[h//4:3*h//4, w//4:3*w//4])
    features['freq_radial_peak'] = np.argmax(radial_profile[1:]) + 1
    
    return features


def extract_structure_features(img_gray):
    features = {}
    gabor_responses = []
    for theta in [0, 45, 90, 135]:
        kernel = cv2.getGaborKernel((21, 21), 5.0, np.radians(theta), 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
        gabor_responses.append(filtered.mean())
    
    features['structure_gabor_0deg'] = gabor_responses[0]
    features['structure_gabor_45deg'] = gabor_responses[1]
    features['structure_gabor_90deg'] = gabor_responses[2]
    features['structure_gabor_135deg'] = gabor_responses[3]
    features['structure_dominant_orientation'] = np.argmax(gabor_responses) * 45
    features['structure_regularity'] = 1.0 / (1.0 + np.std(gabor_responses) / (np.mean(gabor_responses) + 1e-8))
    
    return features


def extract_edge_features(img_gray):
    features = {}
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx)
    
    features['edge_gradient_mean'] = gradient_magnitude.mean()
    features['edge_gradient_std'] = gradient_magnitude.std()
    features['edge_direction_mean'] = gradient_direction.mean()
    features['edge_direction_std'] = gradient_direction.std()
    
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    features['edge_laplacian_variance'] = laplacian.var()
    
    return features


def extract_color_features(img_bgr):
    features = {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    for i, color in enumerate(['r', 'g', 'b']):
        features[f'color_{color}_mean'] = img_rgb[:, :, i].mean()
        features[f'color_{color}_std'] = img_rgb[:, :, i].std()
    
    for i, channel in enumerate(['h', 's', 'v']):
        features[f'color_{channel}_mean'] = img_hsv[:, :, i].mean()
        features[f'color_{channel}_std'] = img_hsv[:, :, i].std()
    
    for i, channel in enumerate(['l', 'a', 'b_lab']):
        features[f'color_{channel}_mean'] = img_lab[:, :, i].mean()
        features[f'color_{channel}_std'] = img_lab[:, :, i].std()
    
    return features


def create_feature_visualization(img, img_gray, features, img_name):
    """Create comprehensive visualization."""
    h, w = img.shape[:2]
    viz_height = h * 3
    viz_width = w * 3
    viz_canvas = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255
    
    viz_canvas[0:h, 0:w] = img
    cv2.putText(viz_canvas, "1. Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)
    enhanced_colored = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    roi_y1, roi_y2 = h//4, 3*h//4
    roi_x1, roi_x2 = w//4, 3*w//4
    cv2.rectangle(enhanced_colored, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    viz_canvas[0:h, w:2*w] = enhanced_colored
    cv2.putText(viz_canvas, "2. Enhanced + ROI", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    roi = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    clahe_roi = clahe.apply(roi)
    
    kernel_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    edges_h = cv2.filter2D(clahe_roi, -1, kernel_h)
    edges_h = np.abs(edges_h)
    
    profile_h = np.mean(edges_h, axis=1)
    profile_h = signal.savgol_filter(profile_h, window_length=11, polyorder=3)
    peaks_h, _ = signal.find_peaks(profile_h, distance=max(2, len(profile_h)//100), prominence=np.max(profile_h)*0.1)
    
    weft_viz = cv2.cvtColor(clahe_roi, cv2.COLOR_GRAY2BGR)
    for peak in peaks_h:
        cv2.line(weft_viz, (0, peak), (weft_viz.shape[1], peak), (0, 255, 0), 1)
    
    weft_resized = cv2.resize(weft_viz, (w, h))
    viz_canvas[0:h, 2*w:3*w] = weft_resized
    weft_count = features.get('weft_count', 0)
    cv2.putText(viz_canvas, f"3. Weft: {weft_count:.1f}", (2*w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    kernel_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    edges_v = cv2.filter2D(clahe_roi, -1, kernel_v)
    edges_v = np.abs(edges_v)
    
    profile_v = np.mean(edges_v, axis=0)
    profile_v = signal.savgol_filter(profile_v, window_length=11, polyorder=3)
    peaks_v, _ = signal.find_peaks(profile_v, distance=max(2, len(profile_v)//100), prominence=np.max(profile_v)*0.1)
    
    warp_viz = cv2.cvtColor(clahe_roi, cv2.COLOR_GRAY2BGR)
    for peak in peaks_v:
        cv2.line(warp_viz, (peak, 0), (peak, warp_viz.shape[0]), (255, 0, 0), 1)
    
    warp_resized = cv2.resize(warp_viz, (w, h))
    viz_canvas[h:2*h, 0:w] = warp_resized
    warp_count = features.get('warp_count', 0)
    cv2.putText(viz_canvas, f"4. Warp: {warp_count:.1f}", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    grid_viz = cv2.cvtColor(clahe_roi, cv2.COLOR_GRAY2BGR)
    for peak in peaks_h:
        cv2.line(grid_viz, (0, peak), (grid_viz.shape[1], peak), (0, 255, 0), 1)
    for peak in peaks_v:
        cv2.line(grid_viz, (peak, 0), (peak, grid_viz.shape[0]), (255, 0, 0), 1)
    
    grid_resized = cv2.resize(grid_viz, (w, h))
    viz_canvas[h:2*h, w:2*w] = grid_resized
    cv2.putText(viz_canvas, "5. Thread Grid", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    fft = fft2(img_gray)
    fft_shifted = fftshift(fft)
    magnitude = np.abs(fft_shifted)
    magnitude_log = np.log1p(magnitude)
    magnitude_log = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fft_colored = cv2.applyColorMap(magnitude_log, cv2.COLORMAP_JET)
    
    viz_canvas[h:2*h, 2*w:3*w] = fft_colored
    cv2.putText(viz_canvas, "6. FFT", (2*w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(viz_canvas, f"Image: {img_name}", (10, viz_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return viz_canvas


if __name__ == "__main__":
    start_time = time.time()
    
    df_features = extract_all_fabric_features(
        dataset_dir="data/preprocessed_dataset",
        output_dir="data/feature_extracted_dataset",
        save_visualizations=True,
        n_workers=None
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Complete in {elapsed_time:.1f}s!")
    print(f"⚡ Average: {elapsed_time/len(df_features):.2f}s per image")
