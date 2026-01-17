"""
Single image feature extraction for Flask app
"""
import cv2
import numpy as np
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift, fft as fft1d, fftfreq
import warnings

warnings.filterwarnings('ignore')

def extract_thread_count(img_gray, direction='horizontal', window_size=50):
    """Extract thread count features."""
    try:
        features = {}
        h, w = img_gray.shape
        
        if direction == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, 1))
            prefix = 'weft'
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, window_size))
            prefix = 'warp'
        
        # Morphological operations
        opened = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find edges
        edges = cv2.Canny(opened, 50, 150)
        
        # Count peaks
        if direction == 'horizontal':
            profile = edges.sum(axis=0)
        else:
            profile = edges.sum(axis=1)
        
        # Find peaks
        peaks, _ = signal.find_peaks(profile, height=profile.max() * 0.1)
        thread_count = len(peaks)
        
        features[f'{prefix}_count'] = float(thread_count)
        features[f'{prefix}_count_normalized'] = float(thread_count / (w if direction == 'horizontal' else h))
        
        if len(peaks) > 1:
            spacings = np.diff(peaks)
            features[f'{prefix}_spacing_avg'] = float(np.mean(spacings))
            features[f'{prefix}_spacing_std'] = float(np.std(spacings))
        else:
            features[f'{prefix}_spacing_avg'] = 0.0
            features[f'{prefix}_spacing_std'] = 0.0
        
        return features
    except:
        return {}

def extract_yarn_density(img_gray):
    """Extract yarn density features."""
    try:
        features = {}
        
        # Binary threshold
        _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            features['yarn_avg_area'] = float(np.mean(areas))
            features['yarn_std_area'] = float(np.std(areas))
            features['yarn_count'] = float(len(contours))
        else:
            features['yarn_avg_area'] = 0.0
            features['yarn_std_area'] = 0.0
            features['yarn_count'] = 0.0
        
        # Density ratio
        yarn_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        features['yarn_density'] = float(yarn_pixels / total_pixels)
        
        return features
    except:
        return {}

def extract_thread_spacing(img_gray):
    """Extract thread spacing features."""
    try:
        features = {}
        
        # Horizontal spacing
        h_profile = img_gray.mean(axis=0)
        h_edges = np.diff(h_profile)
        h_peaks, _ = signal.find_peaks(np.abs(h_edges), height=np.abs(h_edges).max() * 0.1)
        
        # Vertical spacing
        v_profile = img_gray.mean(axis=1)
        v_edges = np.diff(v_profile)
        v_peaks, _ = signal.find_peaks(np.abs(v_edges), height=np.abs(v_edges).max() * 0.1)
        
        if len(h_peaks) > 1:
            features['horizontal_spacing'] = float(np.mean(np.diff(h_peaks)))
        else:
            features['horizontal_spacing'] = 0.0
        
        if len(v_peaks) > 1:
            features['vertical_spacing'] = float(np.mean(np.diff(v_peaks)))
        else:
            features['vertical_spacing'] = 0.0
        
        return features
    except:
        return {}

def extract_texture_features(img_gray):
    """Extract texture features using Haralick-like approach."""
    try:
        features = {}
        
        # GLCM-like features
        # Energy
        glcm = cv2.filter2D(img_gray, -1, np.array([[1, 0], [0, -1]]))
        features['texture_energy'] = float(np.sum(glcm ** 2))
        
        # Entropy
        hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        features['texture_entropy'] = float(entropy)
        
        # Contrast
        features['texture_contrast'] = float(np.sum((img_gray - img_gray.mean()) ** 2) / (img_gray.shape[0] * img_gray.shape[1]))
        
        # Homogeneity
        glcm_norm = glcm / (1 + np.abs(glcm) + 1e-7)
        features['texture_homogeneity'] = float(np.mean(glcm_norm))
        
        return features
    except:
        return {}

def extract_frequency_features(img_gray):
    """Extract frequency domain features using FFT."""
    try:
        features = {}
        
        # Normalize image
        img_norm = (img_gray - img_gray.mean()) / (img_gray.std() + 1e-7)
        
        # 2D FFT
        fft_result = fft2(img_norm)
        fft_shift = fftshift(fft_result)
        magnitude = np.abs(fft_shift)
        
        # Extract features
        features['fft_magnitude_mean'] = float(np.mean(magnitude))
        features['fft_magnitude_std'] = float(np.std(magnitude))
        features['fft_magnitude_max'] = float(np.max(magnitude))
        
        # Power spectrum
        power = np.abs(fft_shift) ** 2
        features['power_spectrum_mean'] = float(np.mean(power))
        
        return features
    except:
        return {}

def extract_structure_features(img_gray):
    """Extract structural features."""
    try:
        features = {}
        
        # Edge detection
        edges = cv2.Canny(img_gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
        
        # Laplacian
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        features['laplacian_variance'] = float(np.var(laplacian))
        features['laplacian_mean'] = float(np.mean(np.abs(laplacian)))
        
        # Sobel
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        features['sobel_magnitude_mean'] = float(np.mean(sobel_mag))
        features['sobel_magnitude_std'] = float(np.std(sobel_mag))
        
        return features
    except:
        return {}

def extract_edge_features(img_gray):
    """Extract edge-based features."""
    try:
        features = {}
        
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        features['line_count'] = float(len(lines) if lines is not None else 0)
        
        # Morphological properties
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        features['edge_connectivity'] = float(np.sum(opened > 0) / (opened.shape[0] * opened.shape[1]))
        
        return features
    except:
        return {}

def extract_color_features(img):
    """Extract color features."""
    try:
        features = {}
        
        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            
            features['color_r_mean'] = float(np.mean(r))
            features['color_r_std'] = float(np.std(r))
            features['color_g_mean'] = float(np.mean(g))
            features['color_g_std'] = float(np.std(g))
            features['color_b_mean'] = float(np.mean(b))
            features['color_b_std'] = float(np.std(b))
        else:
            features['color_r_mean'] = 0.0
            features['color_r_std'] = 0.0
            features['color_g_mean'] = 0.0
            features['color_g_std'] = 0.0
            features['color_b_mean'] = 0.0
            features['color_b_std'] = 0.0
        
        return features
    except:
        return {}

def extract_all_fabric_features_single_image(img):
    """
    Extract all fabric features from a single image.
    
    Args:
        img: OpenCV image (BGR format)
    
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
        
        # Extract features
        try:
            features.update(extract_thread_count(img_gray, direction='horizontal'))
        except:
            pass
        
        try:
            features.update(extract_thread_count(img_gray, direction='vertical'))
        except:
            pass
        
        try:
            features.update(extract_yarn_density(img_gray))
        except:
            pass
        
        try:
            features.update(extract_thread_spacing(img_gray))
        except:
            pass
        
        try:
            features.update(extract_texture_features(img_gray))
        except:
            pass
        
        try:
            features.update(extract_frequency_features(img_gray))
        except:
            pass
        
        try:
            features.update(extract_structure_features(img_gray))
        except:
            pass
        
        try:
            features.update(extract_edge_features(img_gray))
        except:
            pass
        
        try:
            features.update(extract_color_features(img))
        except:
            pass
        
        return features if features else None
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None
