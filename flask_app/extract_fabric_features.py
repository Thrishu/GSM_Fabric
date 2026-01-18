"""
Complete fabric feature extraction for Flask app - single image version
"""
import numpy as np
import cv2
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift, fft as fft1d, fftfreq
import warnings

warnings.filterwarnings('ignore')

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
        features[f'{prefix}count_autocorr'] = float(autocorr_count)
    else:
        features[f'{prefix}count_autocorr'] = 0.0
    
    if thread_counts_fft:
        fft_count = np.median(thread_counts_fft)
        all_counts.append(fft_count)
        features[f'{prefix}count_fft'] = float(fft_count)
    else:
        features[f'{prefix}count_fft'] = 0.0
    
    if direct_count > 0:
        all_counts.append(float(direct_count))
        features[f'{prefix}count_direct'] = float(direct_count)
    else:
        features[f'{prefix}count_direct'] = 0.0
    
    if all_counts:
        final_count = np.median(all_counts)
        if final_count < 5:
            final_count = max(all_counts) if all_counts else 0
        elif final_count > 300:
            final_count = min(all_counts) if all_counts else 0
    else:
        final_count = 0
    
    features[f'{prefix}count'] = float(final_count)
    
    if spacings:
        features[f'{prefix}spacing_avg'] = float(np.median(spacings))
        features[f'{prefix}spacing_std'] = float(np.std(spacings))
        features[f'{prefix}spacing_cv'] = float(np.std(spacings) / (np.mean(spacings) + 1e-8))
    else:
        features[f'{prefix}spacing_avg'] = 0.0
        features[f'{prefix}spacing_std'] = 0.0
        features[f'{prefix}spacing_cv'] = 0.0
    
    if len(all_counts) > 1:
        count_std = np.std(all_counts)
        count_mean = np.mean(all_counts)
        confidence = 1.0 / (1.0 + count_std / (count_mean + 1e-8))
        features[f'{prefix}confidence'] = float(confidence)
    else:
        features[f'{prefix}confidence'] = 0.5
    
    return features

def extract_yarn_density(img_gray):
    features = {}
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features['yarn_edge_density'] = float(edge_density)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    features['yarn_component_count'] = float(num_labels - 1)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        features['yarn_avg_area'] = float(np.mean(areas))
        features['yarn_std_area'] = float(np.std(areas))
        features['yarn_density_uniformity'] = float(1.0 / (1.0 + np.std(areas) / (np.mean(areas) + 1e-8)))
    else:
        features['yarn_avg_area'] = 0.0
        features['yarn_std_area'] = 0.0
        features['yarn_density_uniformity'] = 0.0
    
    return features

def extract_thread_spacing(img_gray):
    features = {}
    edges = cv2.Canny(img_gray, 50, 150)
    
    lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines_h is not None:
        y_positions = sorted([line[0][1] for line in lines_h])
        h_spacing = np.diff(y_positions) if len(y_positions) > 1 else [0]
        features['weft_spacing_uniformity'] = float(1.0 / (1.0 + np.std(h_spacing) / (np.mean(h_spacing) + 1e-8)))
        features['weft_lines_detected'] = float(len(lines_h))
    else:
        features['weft_spacing_uniformity'] = 0.0
        features['weft_lines_detected'] = 0.0
    
    lines_v = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=50, minLineLength=30, maxLineGap=10)
    if lines_v is not None:
        x_positions = sorted([line[0][0] for line in lines_v])
        v_spacing = np.diff(x_positions) if len(x_positions) > 1 else [0]
        features['warp_spacing_uniformity'] = float(1.0 / (1.0 + np.std(v_spacing) / (np.mean(v_spacing) + 1e-8)))
        features['warp_lines_detected'] = float(len(lines_v))
    else:
        features['warp_spacing_uniformity'] = 0.0
        features['warp_lines_detected'] = 0.0
    
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


def extract_texture_features(img_gray):
    features = {}
    dx = ndimage.sobel(img_gray, axis=1)
    dy = ndimage.sobel(img_gray, axis=0)
    features['texture_contrast'] = float(np.sqrt(dx**2 + dy**2).mean())
    
    local_std = ndimage.generic_filter(img_gray, np.std, size=5)
    features['texture_homogeneity'] = float(1.0 / (1.0 + local_std.mean()))
    
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    features['texture_energy'] = float(np.sum(hist**2))
    
    hist = hist[hist > 0]
    features['texture_entropy'] = float(-np.sum(hist * np.log2(hist)))
    
    lbp = compute_lbp(img_gray)
    features['texture_lbp_mean'] = float(lbp.mean())
    features['texture_lbp_std'] = float(lbp.std())
    
    return features

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
    
    features['freq_dc_component'] = float(magnitude_log[center[0], center[1]])
    features['freq_high_freq_energy'] = float(np.sum(magnitude_log[0:h//4, :]) + np.sum(magnitude_log[3*h//4:, :]))
    features['freq_low_freq_energy'] = float(np.sum(magnitude_log[h//4:3*h//4, w//4:3*w//4]))
    features['freq_radial_peak'] = float(np.argmax(radial_profile[1:]) + 1)
    
    return features

def extract_structure_features(img_gray):
    features = {}
    gabor_responses = []
    for theta in [0, 45, 90, 135]:
        kernel = cv2.getGaborKernel((21, 21), 5.0, np.radians(theta), 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
        gabor_responses.append(float(filtered.mean()))
    
    features['structure_gabor_0deg'] = gabor_responses[0]
    features['structure_gabor_45deg'] = gabor_responses[1]
    features['structure_gabor_90deg'] = gabor_responses[2]
    features['structure_gabor_135deg'] = gabor_responses[3]
    features['structure_dominant_orientation'] = float(np.argmax(gabor_responses) * 45)
    features['structure_regularity'] = float(1.0 / (1.0 + np.std(gabor_responses) / (np.mean(gabor_responses) + 1e-8)))
    
    return features

def extract_edge_features(img_gray):
    features = {}
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx)
    
    features['edge_gradient_mean'] = float(gradient_magnitude.mean())
    features['edge_gradient_std'] = float(gradient_magnitude.std())
    features['edge_direction_mean'] = float(gradient_direction.mean())
    features['edge_direction_std'] = float(gradient_direction.std())
    
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    features['edge_laplacian_variance'] = float(laplacian.var())
    
    return features

def extract_color_features(img_bgr):
    features = {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    for i, color in enumerate(['r', 'g', 'b']):
        features[f'color_{color}_mean'] = float(img_rgb[:, :, i].mean())
        features[f'color_{color}_std'] = float(img_rgb[:, :, i].std())
    
    for i, channel in enumerate(['h', 's', 'v']):
        features[f'color_{channel}_mean'] = float(img_hsv[:, :, i].mean())
        features[f'color_{channel}_std'] = float(img_hsv[:, :, i].std())
    
    for i, channel in enumerate(['l', 'a', 'b_lab']):
        features[f'color_{channel}_mean'] = float(img_lab[:, :, i].mean())
        features[f'color_{channel}_std'] = float(img_lab[:, :, i].std())
    
    return features

def extract_all_fabric_features_single_image(img, debug=False):
    """
    Extract all fabric features from a single image.
    
    Args:
        img: OpenCV image (BGR format)
        debug: Enable debug logging
    
    Returns:
        Dictionary of extracted features
    """
    try:
        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        # Initialize features dictionary
        features = {}
        
        # Extract features with error logging
        try:
            thread_count_h = extract_thread_count(img_gray, direction='horizontal')
            features.update(thread_count_h)
            if debug:
                print(f"[DEBUG] Extracted {len(thread_count_h)} weft features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_thread_count (horizontal) failed: {str(e)}")
        
        try:
            thread_count_v = extract_thread_count(img_gray, direction='vertical')
            features.update(thread_count_v)
            if debug:
                print(f"[DEBUG] Extracted {len(thread_count_v)} warp features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_thread_count (vertical) failed: {str(e)}")
        
        try:
            yarn_density = extract_yarn_density(img_gray)
            features.update(yarn_density)
            if debug:
                print(f"[DEBUG] Extracted {len(yarn_density)} yarn density features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_yarn_density failed: {str(e)}")
        
        try:
            thread_spacing = extract_thread_spacing(img_gray)
            features.update(thread_spacing)
            if debug:
                print(f"[DEBUG] Extracted {len(thread_spacing)} thread spacing features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_thread_spacing failed: {str(e)}")
        
        try:
            texture = extract_texture_features(img_gray)
            features.update(texture)
            if debug:
                print(f"[DEBUG] Extracted {len(texture)} texture features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_texture_features failed: {str(e)}")
        
        try:
            frequency = extract_frequency_features(img_gray)
            features.update(frequency)
            if debug:
                print(f"[DEBUG] Extracted {len(frequency)} frequency features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_frequency_features failed: {str(e)}")
        
        try:
            structure = extract_structure_features(img_gray)
            features.update(structure)
            if debug:
                print(f"[DEBUG] Extracted {len(structure)} structure features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_structure_features failed: {str(e)}")
        
        try:
            edge = extract_edge_features(img_gray)
            features.update(edge)
            if debug:
                print(f"[DEBUG] Extracted {len(edge)} edge features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_edge_features failed: {str(e)}")
        
        try:
            color = extract_color_features(img)
            features.update(color)
            if debug:
                print(f"[DEBUG] Extracted {len(color)} color features")
        except Exception as e:
            if debug:
                print(f"[WARNING] extract_color_features failed: {str(e)}")
        
        if debug:
            print(f"[DEBUG] Total features extracted: {len(features)}")
        
        return features if features else None
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None
